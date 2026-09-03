#!/usr/bin/env bash
set -euo pipefail

# Frozen B6 + TriFusion observer.
#
# This runner is intentionally observer-only.  It replays the immutable SAM3
# teacher cache into three diagnostic branches:
#   M1/M2: missing-instance proposals + incremental Mask Graph;
#   M3:    Top-K Mask-RGBD occupancy/MSR oriented-box proposals;
#   M4:    AP50-aware safety features (and predictions when a checkpoint is
#          supplied).
#
# None of these branches may change exported boxes, scores, count, or order.
#
# Fixed 10 scenes:
#   bash scripts/run_scannet_trifusion_observer.sh 0,1
#
# Full 100 scenes:
#   BOXFUSION_TRIFUSION_FULL100=1 \
#     bash scripts/run_scannet_trifusion_observer.sh 0,1
#
# Optional train-only M4 checkpoint (observer predictions only):
#   BOXFUSION_TRIFUSION_AP50_GATE_CHECKPOINT=/path/ap50_gate.npz \
#     bash scripts/run_scannet_trifusion_observer.sh 0,1
#
# Resume only an unchanged interrupted run:
#   BOXFUSION_TRIFUSION_ALLOW_RESUME=1 \
#   BOXFUSION_TRIFUSION_RUN_TAG=<same-tag> \
#     bash scripts/run_scannet_trifusion_observer.sh 0,1

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FULL100="${BOXFUSION_TRIFUSION_FULL100:-0}"
ALLOW_RESUME="${BOXFUSION_TRIFUSION_ALLOW_RESUME:-0}"
CONFIG="${BOXFUSION_TRIFUSION_CONFIG:-$ROOT/config/scannet_online_refinement.yaml}"
YOLOE_CHECKPOINT="${BOXFUSION_TRIFUSION_YOLOE_CHECKPOINT:-${BOXFUSION_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}}"
QUALITY_CHECKPOINT="${BOXFUSION_TRIFUSION_B6_CHECKPOINT:-${BOXFUSION_QUALITY_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}}"
TEACHER_CACHE="${BOXFUSION_TRIFUSION_TEACHER_CACHE:-$ROOT/cache/sam3_teacher/sam3_teacher_full100_c050_frozen_v1}"
TEACHER_NAMESPACE="${BOXFUSION_TRIFUSION_TEACHER_NAMESPACE:-sam3-scannet18-val100-c050-frozen-v1}"
CACHE_MISSING_POLICY="${BOXFUSION_TRIFUSION_CACHE_MISSING_POLICY:-error}"
AP50_GATE_CHECKPOINT="${BOXFUSION_TRIFUSION_AP50_GATE_CHECKPOINT:-}"

case "$FULL100" in
    0|1) ;;
    *) echo "BOXFUSION_TRIFUSION_FULL100 must be 0 or 1" >&2; exit 2 ;;
esac
case "$ALLOW_RESUME" in
    0|1) ;;
    *) echo "BOXFUSION_TRIFUSION_ALLOW_RESUME must be 0 or 1" >&2; exit 2 ;;
esac
case "$CACHE_MISSING_POLICY" in
    error|empty) ;;
    *)
        echo "BOXFUSION_TRIFUSION_CACHE_MISSING_POLICY must be error or empty" >&2
        exit 2
        ;;
esac

if [[ -n "${BOXFUSION_TRIFUSION_SCENE_LIST:-}" ]]; then
    SCENE_LIST="$BOXFUSION_TRIFUSION_SCENE_LIST"
    SCOPE_TAG="custom"
elif [[ "$FULL100" == "1" ]]; then
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
    SCOPE_TAG="full100"
else
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
    SCOPE_TAG="ablation10"
fi

RUN_TAG="${BOXFUSION_TRIFUSION_RUN_TAG:-trifusion_plus10_observer_${SCOPE_TAG}_v1}"
PRED_ROOT="${BOXFUSION_TRIFUSION_PRED_ROOT:-$ROOT/results/$RUN_TAG}"
LOG_ROOT="${BOXFUSION_TRIFUSION_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
DIAGNOSTICS_ROOT="${BOXFUSION_TRIFUSION_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/$RUN_TAG}"
EVAL_ROOT="${BOXFUSION_TRIFUSION_EVAL_ROOT:-$ROOT/evaluation/$RUN_TAG}"

required_files=(
    "$CONFIG"
    "$SCENE_LIST"
    "$YOLOE_CHECKPOINT"
    "$QUALITY_CHECKPOINT"
)
if [[ -n "$AP50_GATE_CHECKPOINT" ]]; then
    if [[ "${AP50_GATE_CHECKPOINT,,}" != *.npz ]]; then
        echo "BOXFUSION_TRIFUSION_AP50_GATE_CHECKPOINT must end in .npz" >&2
        exit 2
    fi
    required_files+=("$AP50_GATE_CHECKPOINT")
fi
for path in "${required_files[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required TriFusion input: $path" >&2
        exit 1
    fi
done
if [[ ! -s "$SCENE_LIST" ]]; then
    echo "TriFusion scene list is empty: $SCENE_LIST" >&2
    exit 1
fi
if [[ ! -d "$TEACHER_CACHE" ]]; then
    echo "Missing immutable SAM3 teacher cache: $TEACHER_CACHE" >&2
    exit 1
fi
if [[ -z "$TEACHER_NAMESPACE" ]]; then
    echo "BOXFUSION_TRIFUSION_TEACHER_NAMESPACE cannot be empty" >&2
    exit 1
fi

# A fresh tag is mandatory unless this is an explicit resume.  This prevents
# outputs from different profiles, checkpoints, caches, or code revisions
# from being silently mixed.
if [[ "$ALLOW_RESUME" != "1" ]]; then
    for directory in \
        "$PRED_ROOT" "$LOG_ROOT" "$DIAGNOSTICS_ROOT" "$EVAL_ROOT"; do
        if [[ -d "$directory" ]] \
            && [[ -n "$(find "$directory" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            echo "Refusing to reuse non-empty TriFusion directory:" >&2
            echo "  $directory" >&2
            echo "Choose a fresh BOXFUSION_TRIFUSION_RUN_TAG." >&2
            exit 1
        fi
    done
fi

unset BOXFUSION_DISABLE_ONLINE_REFINEMENT
unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_JOINT_CHECKPOINT
unset BOXFUSION_JOINT_DETECTOR_BLEND
unset BOXFUSION_PROPOSAL_CACHE_DIRECTORY
unset BOXFUSION_PROPOSAL_CACHE_NAMESPACE
unset BOXFUSION_PROPOSAL_CACHE_MISSING_POLICY

export BOXFUSION_ONLINE_CONFIG="$CONFIG"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_ONLINE_ABLATION_PROFILE="trifusion_plus10_observer"

# Exact frozen B6 anchor.
export BOXFUSION_PROPOSAL_PROVIDER="yoloe"
export BOXFUSION_YOLOE_CHECKPOINT="$YOLOE_CHECKPOINT"
export BOXFUSION_QUALITY_CHECKPOINT="$QUALITY_CHECKPOINT"
export BOXFUSION_QUALITY_MODE="iou_mlp"
export BOXFUSION_QUALITY_DETECTOR_BLEND="0.40"
export BOXFUSION_SCANNET_MIN_EXTENT="0.40"
export BOXFUSION_PROPOSAL_INTERVAL="5"
export BOXFUSION_CANDIDATE_TTL_CLOCK="provider_call"
export BOXFUSION_CANDIDATE_TRACK_TTL="3"
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS="0"
export BOXFUSION_INFERENCE_SEED="0"
export BOXFUSION_EVAL_SEED="0"

# Read-only SAM3 teacher stream.  The generic C4 environment variables are
# reused by the core cache-only provider, but the profile selects the new
# TriFusion diagnostic branches.
export BOXFUSION_C4_PROPOSAL_CACHE_DIRECTORY="$TEACHER_CACHE"
export BOXFUSION_C4_PROPOSAL_CACHE_NAMESPACE="$TEACHER_NAMESPACE"
export BOXFUSION_C4_PROPOSAL_CACHE_MISSING_POLICY="$CACHE_MISSING_POLICY"
if [[ -n "$AP50_GATE_CHECKPOINT" ]]; then
    export BOXFUSION_TRIFUSION_GATE_CHECKPOINT="$AP50_GATE_CHECKPOINT"
else
    # Do not let a stale generic runner variable silently change an
    # observer-only run which did not explicitly request M4 inference.
    unset BOXFUSION_TRIFUSION_GATE_CHECKPOINT
fi

export BOXFUSION_ONLINE_PRED_ROOT="$PRED_ROOT"
export BOXFUSION_ONLINE_LOG_ROOT="$LOG_ROOT"
export BOXFUSION_DIAGNOSTICS_ROOT="$DIAGNOSTICS_ROOT"
export BOXFUSION_EVAL_ROOT="$EVAL_ROOT"

scene_count="$(awk 'NF { count += 1 } END { print count + 0 }' "$SCENE_LIST")"
echo "Frozen B6 + TriFusion multi-module observer"
echo "  profile: trifusion_plus10_observer"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  frozen B6: detector blend=0.40 / minimum extent=0.40"
echo "  teacher: cache_only / $TEACHER_CACHE"
echo "  modules: M1 missing proposals + M2 incremental graph + M3 occupancy/MSR + M4 AP50 features"
echo "  M4 AP50 gate checkpoint: ${AP50_GATE_CHECKPOINT:-disabled (features only)}"
echo "  output contract: observer identity (boxes/scores/count/order unchanged)"
echo "  run tag: $RUN_TAG"
echo "  GPUs: $GPU_SPEC"

exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
