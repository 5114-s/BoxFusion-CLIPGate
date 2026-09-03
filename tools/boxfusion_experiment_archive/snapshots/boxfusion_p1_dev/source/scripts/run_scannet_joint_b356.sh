#!/usr/bin/env bash
set -euo pipefail

# Evaluate the isolated B3 -> B5 + B6-v2 joint local head.
#
# This wrapper does not train and never reuses the legacy B3/B5/B6 result
# namespaces.  By default it runs the fixed ten-scene validation ablation; set
# BOXFUSION_JOINT_B356_SCENE_LIST explicitly before a later 100-scene run.
#
# Usage (run manually after the current experiment has finished):
#   bash scripts/run_scannet_joint_b356.sh 0,1

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT="${BOXFUSION_JOINT_B356_CHECKPOINT:-$ROOT/models/scannet_joint_b356_k5_p128_v1.pt}"
SCENE_LIST="${BOXFUSION_JOINT_B356_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
RUN_TAG="${BOXFUSION_JOINT_B356_RUN_TAG:-joint_b356_k5_p128_blend040_ablation10_v1}"
MIN_EXTENT="${BOXFUSION_JOINT_B356_MIN_EXTENT:-0.40}"
ALLOW_RESUME="${BOXFUSION_JOINT_B356_ALLOW_RESUME:-0}"

PRED_ROOT="${BOXFUSION_JOINT_B356_PRED_ROOT:-$ROOT/results/$RUN_TAG}"
LOG_ROOT="${BOXFUSION_JOINT_B356_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
DIAGNOSTICS_ROOT="${BOXFUSION_JOINT_B356_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/$RUN_TAG}"
EVAL_ROOT="${BOXFUSION_JOINT_B356_EVAL_ROOT:-$ROOT/evaluation/$RUN_TAG}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing trained joint B3/B5/B6-v2 checkpoint: $CHECKPOINT" >&2
    echo "Train it first with: bash scripts/train_scannet_joint_b356.sh" >&2
    exit 1
fi
if [[ ! -s "$SCENE_LIST" ]]; then
    echo "Missing or empty evaluation scene list: $SCENE_LIST" >&2
    exit 1
fi
if [[ "$ALLOW_RESUME" != "0" && "$ALLOW_RESUME" != "1" ]]; then
    echo "BOXFUSION_JOINT_B356_ALLOW_RESUME must be 0 or 1" >&2
    exit 1
fi
if [[ ! "$MIN_EXTENT" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "BOXFUSION_JOINT_B356_MIN_EXTENT must be non-negative" >&2
    exit 1
fi

# The lower-level driver is resume-friendly.  This experiment wrapper is
# deliberately stricter: an existing namespace is accepted only after the
# caller explicitly confirms this is an unchanged interrupted run.
if [[ "$ALLOW_RESUME" != "1" ]]; then
    for directory in "$PRED_ROOT" "$LOG_ROOT" "$DIAGNOSTICS_ROOT" "$EVAL_ROOT"; do
        if [[ -d "$directory" ]] \
            && [[ -n "$(find "$directory" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            echo "Refusing to reuse non-empty joint experiment directory:" >&2
            echo "  $directory" >&2
            echo "Choose a fresh BOXFUSION_JOINT_B356_RUN_TAG." >&2
            echo "For an unchanged interrupted run only, explicitly set" >&2
            echo "BOXFUSION_JOINT_B356_ALLOW_RESUME=1." >&2
            exit 1
        fi
    done
fi

unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_QUALITY_CHECKPOINT
unset BOXFUSION_QUALITY_MODE
unset BOXFUSION_QUALITY_DETECTOR_BLEND
unset BOXFUSION_DISABLE_ONLINE_REFINEMENT

export BOXFUSION_JOINT_CHECKPOINT="$CHECKPOINT"
export BOXFUSION_JOINT_DETECTOR_BLEND="${BOXFUSION_JOINT_B356_DETECTOR_BLEND:-0.40}"
export BOXFUSION_ONLINE_CONFIG="$ROOT/config/scannet_online_refinement.yaml"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_ONLINE_ABLATION_PROFILE="joint_b3_b5_b6v2"
export BOXFUSION_PROPOSAL_INTERVAL="${BOXFUSION_JOINT_B356_PROPOSAL_INTERVAL:-5}"
export BOXFUSION_CANDIDATE_TTL_CLOCK="provider_call"
export BOXFUSION_CANDIDATE_TRACK_TTL="3"
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS="0"
export BOXFUSION_SCANNET_MIN_EXTENT="$MIN_EXTENT"
export BOXFUSION_INFERENCE_SEED="0"
export BOXFUSION_EVAL_SEED="0"

export BOXFUSION_ONLINE_PRED_ROOT="$PRED_ROOT"
export BOXFUSION_ONLINE_LOG_ROOT="$LOG_ROOT"
export BOXFUSION_DIAGNOSTICS_ROOT="$DIAGNOSTICS_ROOT"
export BOXFUSION_EVAL_ROOT="$EVAL_ROOT"

scene_count="$(awk 'NF { count += 1 } END { print count + 0 }' "$SCENE_LIST")"
echo "Joint B3 -> B5 + B6-v2 evaluation"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  profile: joint_b3_b5_b6v2"
echo "  checkpoint: $CHECKPOINT"
echo "  detector blend: $BOXFUSION_JOINT_DETECTOR_BLEND"
echo "  minimum extent: $MIN_EXTENT"
echo "  tag: $RUN_TAG"
echo "  GPUs: $GPU_SPEC"
echo "Launching the fixed-scene joint evaluation now."

exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
