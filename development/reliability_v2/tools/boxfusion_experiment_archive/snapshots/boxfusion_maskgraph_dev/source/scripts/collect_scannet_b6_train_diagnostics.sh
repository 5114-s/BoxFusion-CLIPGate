#!/usr/bin/env bash
set -euo pipefail

# Generate no-op B6 supervision diagnostics on ScanNet TRAIN scenes.
# quality_observer keeps masked appearance evidence but never changes boxes,
# scores, or counts.  Use a train-only scene list and a separate frames root.
#
# Usage:
#   BOXFUSION_B6_TRAIN_SCENES=/path/train_subset.txt \
#   BOXFUSION_SCANNET_FRAMES_ROOT=/path/prepared_train_frames \
#     bash scripts/collect_scannet_b6_train_diagnostics.sh 0,1

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENE_LIST="${BOXFUSION_B6_TRAIN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
FRAMES_ROOT="${BOXFUSION_SCANNET_FRAMES_ROOT:-$ROOT/data/scannet_train}"
RUN_TAG="${BOXFUSION_B6_COLLECT_TAG:-b6_quality_observer_train}"

if [[ -z "$SCENE_LIST" || ! -f "$SCENE_LIST" ]]; then
    echo "Missing ScanNet train-only list: $SCENE_LIST" >&2
    exit 1
fi
if [[ "$(basename "$SCENE_LIST")" == *val* ]]; then
    echo "Refusing validation-labelled diagnostics for B6 training" >&2
    exit 1
fi
if [[ -z "$FRAMES_ROOT" || ! -d "$FRAMES_ROOT" ]]; then
    echo "Missing prepared train RGB-D frames: $FRAMES_ROOT" >&2
    echo "Run: bash scripts/prepare_scannet_b6_train_data.sh" >&2
    exit 1
fi

unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_QUALITY_CHECKPOINT
unset BOXFUSION_DISABLE_ONLINE_REFINEMENT
export BOXFUSION_ONLINE_CONFIG="$ROOT/config/scannet_online_refinement.yaml"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_SCANNET_FRAMES_ROOT="$FRAMES_ROOT"
export BOXFUSION_ONLINE_ABLATION_PROFILE="quality_observer"
export BOXFUSION_PROPOSAL_INTERVAL="${BOXFUSION_B6_PROPOSAL_INTERVAL:-5}"
export BOXFUSION_CANDIDATE_TTL_CLOCK="provider_call"
export BOXFUSION_CANDIDATE_TRACK_TTL="3"
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS="0"
export BOXFUSION_SCANNET_MIN_EXTENT="${BOXFUSION_B6_COLLECT_MIN_EXTENT:-0.0}"
export BOXFUSION_INFERENCE_SEED="0"
export BOXFUSION_EVAL_SEED="0"

export BOXFUSION_ONLINE_PRED_ROOT="${BOXFUSION_B6_COLLECT_PRED_ROOT:-$ROOT/results/$RUN_TAG}"
export BOXFUSION_ONLINE_LOG_ROOT="${BOXFUSION_B6_COLLECT_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
export BOXFUSION_DIAGNOSTICS_ROOT="${BOXFUSION_B6_TRAIN_DIAGNOSTICS:-$ROOT/diagnostics/$RUN_TAG}"
export BOXFUSION_EVAL_ROOT="${BOXFUSION_B6_COLLECT_EVAL_ROOT:-$ROOT/evaluation/$RUN_TAG}"

echo "B6 train diagnostics: tag=$RUN_TAG, scenes=$SCENE_LIST, GPUs=$GPU_SPEC"
exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
