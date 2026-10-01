#!/usr/bin/env bash
set -euo pipefail

# B3+B6 ablation: conservative Top-K Mask-RGBD geometry refit plus learned
# IoU-aware quality ranking.  Supplemental output, BoxRefiner, and Soft-NMS
# remain disabled by the b3_b6 profile.
#
# Usage:
#   bash scripts/run_scannet_b3_b6.sh 0,1

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT="${BOXFUSION_QUALITY_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
RUN_TAG="${BOXFUSION_B3_RUN_TAG:-b3_b6_blend040_extent040_ablation10}"
SCENE_LIST="${BOXFUSION_B3_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Missing trained B6 iou_mlp checkpoint: $CHECKPOINT" >&2
    exit 1
fi

unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_DISABLE_ONLINE_REFINEMENT

export BOXFUSION_ONLINE_CONFIG="$ROOT/config/scannet_online_refinement.yaml"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_ONLINE_ABLATION_PROFILE="b3_b6"
export BOXFUSION_QUALITY_MODE="iou_mlp"
export BOXFUSION_QUALITY_CHECKPOINT="$CHECKPOINT"
export BOXFUSION_QUALITY_DETECTOR_BLEND="${BOXFUSION_B3_B6_DETECTOR_BLEND:-0.40}"
export BOXFUSION_SCANNET_MIN_EXTENT="${BOXFUSION_B3_MIN_EXTENT:-0.40}"
export BOXFUSION_PROPOSAL_INTERVAL="${BOXFUSION_B3_PROPOSAL_INTERVAL:-5}"
export BOXFUSION_CANDIDATE_TTL_CLOCK="provider_call"
export BOXFUSION_CANDIDATE_TRACK_TTL="3"
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS="0"
export BOXFUSION_INFERENCE_SEED="0"
export BOXFUSION_EVAL_SEED="0"

export BOXFUSION_ONLINE_PRED_ROOT="${BOXFUSION_B3_PRED_ROOT:-$ROOT/results/$RUN_TAG}"
export BOXFUSION_ONLINE_LOG_ROOT="${BOXFUSION_B3_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
export BOXFUSION_DIAGNOSTICS_ROOT="${BOXFUSION_B3_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/$RUN_TAG}"
export BOXFUSION_EVAL_ROOT="${BOXFUSION_B3_EVAL_ROOT:-$ROOT/evaluation/$RUN_TAG}"

echo "B3+B6 run: tag=$RUN_TAG, scenes=$SCENE_LIST, GPUs=$GPU_SPEC"
exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
