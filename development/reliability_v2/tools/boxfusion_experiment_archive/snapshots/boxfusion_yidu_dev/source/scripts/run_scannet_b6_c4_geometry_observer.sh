#!/usr/bin/env bash
set -euo pipefail

# Frozen-B6 + C4 SAM3 Mask-RGBD oriented local-geometry observer v2.
#
# C4 replays a frozen SAM3 cache as a second, read-only evidence stream.  The
# primary online proposal provider remains YOLOE and C4 is observer-only, so
# this runner is deliberately separate from every missing_mask_graph profile.
#
# Fixed 10-scene diagnostic:
#   bash scripts/run_scannet_b6_c4_geometry_observer.sh 0,1
#
# Full 100-scene diagnostic:
#   BOXFUSION_C4_FULL100=1 \
#     bash scripts/run_scannet_b6_c4_geometry_observer.sh 0,1
#
# Resume only an unchanged interrupted run:
#   BOXFUSION_C4_ALLOW_RESUME=1 BOXFUSION_C4_RUN_TAG=<same-tag> \
#     bash scripts/run_scannet_b6_c4_geometry_observer.sh 0,1

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FULL100="${BOXFUSION_C4_FULL100:-0}"
ALLOW_RESUME="${BOXFUSION_C4_ALLOW_RESUME:-0}"
CONFIG="${BOXFUSION_C4_CONFIG:-$ROOT/config/scannet_online_refinement.yaml}"
YOLOE_CHECKPOINT="${BOXFUSION_C4_YOLOE_CHECKPOINT:-${BOXFUSION_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}}"
QUALITY_CHECKPOINT="${BOXFUSION_C4_B6_CHECKPOINT:-${BOXFUSION_QUALITY_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}}"
C4_CACHE_DIRECTORY="${BOXFUSION_C4_PROPOSAL_CACHE_DIRECTORY:-$ROOT/cache/sam3_teacher/sam3_teacher_full100_c050_frozen_v1}"
C4_CACHE_NAMESPACE="${BOXFUSION_C4_PROPOSAL_CACHE_NAMESPACE:-sam3-scannet18-val100-c050-frozen-v1}"
C4_CACHE_MISSING_POLICY="${BOXFUSION_C4_PROPOSAL_CACHE_MISSING_POLICY:-error}"

if [[ "$FULL100" != "0" && "$FULL100" != "1" ]]; then
    echo "BOXFUSION_C4_FULL100 must be 0 or 1" >&2
    exit 2
fi
if [[ "$ALLOW_RESUME" != "0" && "$ALLOW_RESUME" != "1" ]]; then
    echo "BOXFUSION_C4_ALLOW_RESUME must be 0 or 1" >&2
    exit 2
fi
if [[ "$C4_CACHE_MISSING_POLICY" != "error" \
      && "$C4_CACHE_MISSING_POLICY" != "empty" ]]; then
    echo "BOXFUSION_C4_PROPOSAL_CACHE_MISSING_POLICY must be error or empty" >&2
    exit 2
fi

if [[ -n "${BOXFUSION_C4_SCENE_LIST:-}" ]]; then
    SCENE_LIST="$BOXFUSION_C4_SCENE_LIST"
    SCOPE_TAG="custom"
elif [[ "$FULL100" == "1" ]]; then
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
    SCOPE_TAG="full100"
else
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
    SCOPE_TAG="ablation10"
fi

RUN_TAG="${BOXFUSION_C4_RUN_TAG:-b6_c4_mask_rgbd_oriented_observer_${SCOPE_TAG}_v2}"
PRED_ROOT="${BOXFUSION_C4_PRED_ROOT:-$ROOT/results/$RUN_TAG}"
LOG_ROOT="${BOXFUSION_C4_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
DIAGNOSTICS_ROOT="${BOXFUSION_C4_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/$RUN_TAG}"
EVAL_ROOT="${BOXFUSION_C4_EVAL_ROOT:-$ROOT/evaluation/$RUN_TAG}"

required_files=(
    "$CONFIG"
    "$SCENE_LIST"
    "$YOLOE_CHECKPOINT"
    "$QUALITY_CHECKPOINT"
)
for path in "${required_files[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required C4 input: $path" >&2
        exit 1
    fi
done
if [[ ! -s "$SCENE_LIST" ]]; then
    echo "C4 scene list is empty: $SCENE_LIST" >&2
    exit 1
fi
if [[ ! -d "$C4_CACHE_DIRECTORY" ]]; then
    echo "Missing frozen SAM3 cache directory: $C4_CACHE_DIRECTORY" >&2
    exit 1
fi
if [[ -z "$C4_CACHE_NAMESPACE" ]]; then
    echo "BOXFUSION_C4_PROPOSAL_CACHE_NAMESPACE cannot be empty" >&2
    exit 1
fi

# Never silently combine predictions or diagnostics from different code,
# caches, profiles, or thresholds.
if [[ "$ALLOW_RESUME" != "1" ]]; then
    for directory in \
        "$PRED_ROOT" "$LOG_ROOT" "$DIAGNOSTICS_ROOT" "$EVAL_ROOT"; do
        if [[ -d "$directory" ]] \
            && [[ -n "$(find "$directory" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            echo "Refusing to reuse non-empty C4 experiment directory:" >&2
            echo "  $directory" >&2
            echo "Choose a fresh BOXFUSION_C4_RUN_TAG." >&2
            echo "For an unchanged interrupted run only, set" >&2
            echo "BOXFUSION_C4_ALLOW_RESUME=1." >&2
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
export BOXFUSION_ONLINE_ABLATION_PROFILE="b6_c4_mask_rgbd_observer"

# Primary stream: the same online YOLOE/B6 contract used by the frozen anchor.
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

# Secondary stream: immutable SAM3 teacher cache, never an online SAM3 model.
export BOXFUSION_C4_PROPOSAL_CACHE_DIRECTORY="$C4_CACHE_DIRECTORY"
export BOXFUSION_C4_PROPOSAL_CACHE_NAMESPACE="$C4_CACHE_NAMESPACE"
export BOXFUSION_C4_PROPOSAL_CACHE_MISSING_POLICY="$C4_CACHE_MISSING_POLICY"

export BOXFUSION_ONLINE_PRED_ROOT="$PRED_ROOT"
export BOXFUSION_ONLINE_LOG_ROOT="$LOG_ROOT"
export BOXFUSION_DIAGNOSTICS_ROOT="$DIAGNOSTICS_ROOT"
export BOXFUSION_EVAL_ROOT="$EVAL_ROOT"

scene_count="$(awk 'NF { count += 1 } END { print count + 0 }' "$SCENE_LIST")"
echo "Frozen B6 + C4 SAM3 Mask-RGBD oriented geometry observer v2"
echo "  profile: b6_c4_mask_rgbd_observer"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  primary provider/checkpoint: yoloe / $YOLOE_CHECKPOINT"
echo "  frozen B6 checkpoint: $QUALITY_CHECKPOINT"
echo "  B6 score/min extent: blend=0.40 / extent=0.40"
echo "  schedule/lifecycle: interval=5 / provider_call / ttl=3 / archive=0"
echo "  C4 secondary provider: cache_only (read-only)"
echo "  C4 cache: $C4_CACHE_DIRECTORY"
echo "  C4 namespace: $C4_CACHE_NAMESPACE"
echo "  C4 missing policy: $C4_CACHE_MISSING_POLICY"
echo "  C4 diagnostics: oriented corners preserve yaw; 6D boxes are diagnostic-only"
echo "  run tag: $RUN_TAG"
echo "  GPUs: $GPU_SPEC"

exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
