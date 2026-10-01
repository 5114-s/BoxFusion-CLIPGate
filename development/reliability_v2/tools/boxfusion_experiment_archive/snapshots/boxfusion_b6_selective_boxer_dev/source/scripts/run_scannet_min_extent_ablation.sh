#!/usr/bin/env bash
set -euo pipefail

# Matched B0 diagnostic for ScanNet's hard minimum box extent.
# Runs the same deterministic fixed scene list with online refinement disabled;
# only the final 0.30/0.20/0.15-m extent threshold changes.
#
# Usage:
#   bash scripts/run_scannet_min_extent_ablation.sh 0,1

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENE_LIST="${BOXFUSION_EXTENT_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
EXTENTS="${BOXFUSION_EXTENT_VALUES:-0.30 0.20 0.15}"

unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_QUALITY_CHECKPOINT
unset BOXFUSION_ONLINE_ABLATION_PROFILE
for extent in $EXTENTS; do
    tag="b0_min_extent_${extent/./p}_ablation10"
    export BOXFUSION_ONLINE_CONFIG="$ROOT/config/scannet_online_refinement.yaml"
    export BOXFUSION_SCENE_LIST="$SCENE_LIST"
    export BOXFUSION_DISABLE_ONLINE_REFINEMENT="1"
    export BOXFUSION_SCANNET_MIN_EXTENT="$extent"
    export BOXFUSION_INFERENCE_SEED="0"
    export BOXFUSION_EVAL_SEED="0"
    export BOXFUSION_ONLINE_PRED_ROOT="$ROOT/results/$tag"
    export BOXFUSION_ONLINE_LOG_ROOT="$ROOT/logs/$tag"
    export BOXFUSION_DIAGNOSTICS_ROOT="$ROOT/diagnostics/$tag"
    export BOXFUSION_EVAL_ROOT="$ROOT/evaluation/$tag"
    echo "Minimum-extent diagnostic: threshold=$extent, tag=$tag"
    bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
done
