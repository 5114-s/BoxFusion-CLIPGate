#!/usr/bin/env bash
set -euo pipefail

# Fixed-10 B5-v3 AP50-aware ablation.  It uses the same K=5 runtime memory and
# safety gates as B5-v2, changes geometry only, and preserves detector scores,
# detection count/order, and OBB orientation.
#
# Usage:
#   bash scripts/run_scannet_b5v3_ap50_refiner.sh 0,1

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT="${BOXFUSION_B5V3_AP50_CHECKPOINT:-$ROOT/models/scannet_b5v3_k5_ap50_gatealigned_refiner_v2.pt}"
RUN_TAG="${BOXFUSION_B5V3_RUN_TAG:-b5v3_ap50_gatealigned_refiner_only_extent040_ablation10_v2}"
SCENE_LIST="${BOXFUSION_B5V3_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing B5-v3 AP50-aware checkpoint: $CHECKPOINT" >&2
    echo "Run: bash scripts/train_scannet_b5v3_ap50_refiner.sh" >&2
    exit 1
fi
if [[ ! -f "$SCENE_LIST" ]]; then
    echo "Missing B5-v3 evaluation scene list: $SCENE_LIST" >&2
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
export BOXFUSION_SCANNET_MIN_EXTENT="${BOXFUSION_B5V3_RUNTIME_MIN_EXTENT:-0.40}"
export BOXFUSION_PROPOSAL_INTERVAL="${BOXFUSION_B5V3_PROPOSAL_INTERVAL:-5}"
export BOXFUSION_CANDIDATE_TTL_CLOCK="provider_call"
export BOXFUSION_CANDIDATE_TRACK_TTL="3"
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS="0"
export BOXFUSION_INFERENCE_SEED="0"
export BOXFUSION_EVAL_SEED="0"

export BOXFUSION_ONLINE_PRED_ROOT="${BOXFUSION_B5V3_PRED_ROOT:-$ROOT/results/$RUN_TAG}"
export BOXFUSION_ONLINE_LOG_ROOT="${BOXFUSION_B5V3_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
export BOXFUSION_DIAGNOSTICS_ROOT="${BOXFUSION_B5V3_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/$RUN_TAG}"
export BOXFUSION_EVAL_ROOT="${BOXFUSION_B5V3_EVAL_ROOT:-$ROOT/evaluation/$RUN_TAG}"

echo "B5-v3 AP50-aware refiner-only evaluation"
echo "  checkpoint: $CHECKPOINT"
echo "  scenes: $SCENE_LIST"
echo "  tag: $RUN_TAG"
echo "  profile: b5v2_refiner_only (K=5, geometry only)"
echo "  minimum extent: $BOXFUSION_SCANNET_MIN_EXTENT"

exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
