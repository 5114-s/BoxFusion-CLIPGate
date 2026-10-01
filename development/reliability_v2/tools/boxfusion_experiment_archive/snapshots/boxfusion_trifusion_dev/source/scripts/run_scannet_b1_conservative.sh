#!/usr/bin/env bash
set -euo pipefail

# Reproducible B1 conservative-gate ablation.
#
# Fixed model settings:
#   - provider-call candidate clock, TTL=3
#   - confirmed-track archive disabled
#   - supplemental score >= 0.25
#   - weighted multi-view projection IoU >= 0.30
#   - drop supplemental boxes at global 3D IoU >= 0.30
#   - ignore/output-filter global boxes with any extent < 0.30 m
#
# Usage:
#   bash scripts/run_scannet_b1_conservative.sh 0,1
#
# The default scene list is the fixed deterministic 10-scene split.  Override
# BOXFUSION_B1_SCENE_LIST and BOXFUSION_B1_RUN_TAG for another isolated run.

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_TAG="${BOXFUSION_B1_RUN_TAG:-b1_conservative_pj03_g03_s025_ablation10}"
SCENE_LIST="${BOXFUSION_B1_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"

export BOXFUSION_ONLINE_CONFIG="$ROOT/config/scannet_online_refinement.yaml"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_ONLINE_ABLATION_PROFILE="supplemental_conservative"
export BOXFUSION_PROPOSAL_INTERVAL="5"
export BOXFUSION_CANDIDATE_TTL_CLOCK="provider_call"
export BOXFUSION_CANDIDATE_TRACK_TTL="3"
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS="0"
export BOXFUSION_INFERENCE_SEED="0"
export BOXFUSION_EVAL_SEED="0"

# Use B1-specific overrides so stale generic variables from an earlier
# experiment cannot silently redirect this run into an existing directory.
export BOXFUSION_ONLINE_PRED_ROOT="${BOXFUSION_B1_PRED_ROOT:-$ROOT/results/$RUN_TAG}"
export BOXFUSION_ONLINE_LOG_ROOT="${BOXFUSION_B1_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
export BOXFUSION_DIAGNOSTICS_ROOT="${BOXFUSION_B1_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/$RUN_TAG}"
export BOXFUSION_EVAL_ROOT="${BOXFUSION_B1_EVAL_ROOT:-$ROOT/evaluation/$RUN_TAG}"

echo "B1 conservative run: tag=$RUN_TAG, scenes=$SCENE_LIST, GPUs=$GPU_SPEC"
exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
