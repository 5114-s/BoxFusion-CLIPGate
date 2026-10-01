#!/usr/bin/env bash
set -euo pipefail

# Strict B5-v2 + B6 ablation. B5-v2 changes only object-local geometry. B6
# ranks detections with original-geometry features and an explicit detector
# blend (0.40 by default). Supplemental output and Soft-NMS stay disabled.
#
# Usage:
#   bash scripts/run_scannet_b5v2_b6.sh 0,1

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REFINER_CHECKPOINT="${BOXFUSION_B5V2_REFINER_CHECKPOINT:-${BOXFUSION_REFINER_CHECKPOINT:-$ROOT/models/scannet_b5v2_oriented_refiner.pt}}"
QUALITY_CHECKPOINT="${BOXFUSION_B5V2_QUALITY_CHECKPOINT:-${BOXFUSION_QUALITY_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}}"
RUN_TAG="${BOXFUSION_B5V2_RUN_TAG:-b5v2_b6_blend040_extent040_ablation10}"
SCENE_LIST="${BOXFUSION_B5V2_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"

if [[ -z "$REFINER_CHECKPOINT" || ! -f "$REFINER_CHECKPOINT" ]]; then
    echo "Missing trained B5-v2 BoxRefiner checkpoint: $REFINER_CHECKPOINT" >&2
    echo "Run: bash scripts/train_scannet_b5v2_refiner.sh" >&2
    exit 1
fi
if [[ -z "$QUALITY_CHECKPOINT" || ! -f "$QUALITY_CHECKPOINT" ]]; then
    echo "Missing trained B6 iou_mlp checkpoint: $QUALITY_CHECKPOINT" >&2
    echo "Run: bash scripts/train_scannet_b6_quality.sh" >&2
    exit 1
fi

unset BOXFUSION_DISABLE_ONLINE_REFINEMENT

export BOXFUSION_REFINER_CHECKPOINT="$REFINER_CHECKPOINT"
export BOXFUSION_QUALITY_CHECKPOINT="$QUALITY_CHECKPOINT"
export BOXFUSION_QUALITY_MODE="iou_mlp"
export BOXFUSION_QUALITY_DETECTOR_BLEND="${BOXFUSION_B5V2_B6_DETECTOR_BLEND:-0.40}"
export BOXFUSION_ONLINE_CONFIG="$ROOT/config/scannet_online_refinement.yaml"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_ONLINE_ABLATION_PROFILE="b5v2_b6"
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

echo "B5-v2+B6: tag=$RUN_TAG, scenes=$SCENE_LIST, blend=$BOXFUSION_QUALITY_DETECTOR_BLEND, GPUs=$GPU_SPEC"
exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
