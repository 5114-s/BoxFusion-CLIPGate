#!/usr/bin/env bash
set -euo pipefail

# Fixed-10 identity control for the exact B5-v2 runtime contract. It builds
# K=5 memory and diagnostics but leaves boxes, scores, order, and count
# untouched.

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_TAG="${BOXFUSION_B5V3_IDENTITY_RUN_TAG:-b5v3_gatealigned_k5_identity_extent040_ablation10_v2}"
SCENE_LIST="${BOXFUSION_B5V3_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"

unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_QUALITY_CHECKPOINT
unset BOXFUSION_QUALITY_MODE
unset BOXFUSION_QUALITY_DETECTOR_BLEND
unset BOXFUSION_DISABLE_ONLINE_REFINEMENT

export BOXFUSION_ONLINE_CONFIG="$ROOT/config/scannet_online_refinement.yaml"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_ONLINE_ABLATION_PROFILE="b5v2_memory_observer"
export BOXFUSION_SCANNET_MIN_EXTENT="${BOXFUSION_B5V3_RUNTIME_MIN_EXTENT:-0.40}"
export BOXFUSION_PROPOSAL_INTERVAL="${BOXFUSION_B5V3_PROPOSAL_INTERVAL:-5}"
export BOXFUSION_CANDIDATE_TTL_CLOCK="provider_call"
export BOXFUSION_CANDIDATE_TRACK_TTL="3"
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS="0"
export BOXFUSION_INFERENCE_SEED="0"
export BOXFUSION_EVAL_SEED="0"

export BOXFUSION_ONLINE_PRED_ROOT="${BOXFUSION_B5V3_IDENTITY_PRED_ROOT:-$ROOT/results/$RUN_TAG}"
export BOXFUSION_ONLINE_LOG_ROOT="${BOXFUSION_B5V3_IDENTITY_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
export BOXFUSION_DIAGNOSTICS_ROOT="${BOXFUSION_B5V3_IDENTITY_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/$RUN_TAG}"
export BOXFUSION_EVAL_ROOT="${BOXFUSION_B5V3_IDENTITY_EVAL_ROOT:-$ROOT/evaluation/$RUN_TAG}"

echo "B5-v3 gate-aligned identity"
echo "  scenes: $SCENE_LIST"
echo "  tag: $RUN_TAG"
echo "  profile: b5v2_memory_observer"
echo "  GPUs: $GPU_SPEC"

exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
