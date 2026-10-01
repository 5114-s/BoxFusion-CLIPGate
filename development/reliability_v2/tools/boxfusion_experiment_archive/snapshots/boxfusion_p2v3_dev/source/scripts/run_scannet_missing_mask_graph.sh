#!/usr/bin/env bash
set -euo pipefail

# Missing-track identity + incremental Mask Graph evaluation.
#
# Usage:
#   bash scripts/run_scannet_missing_mask_graph.sh 0,1 observer
#   bash scripts/run_scannet_missing_mask_graph.sh 0,1 supplemental
#   bash scripts/run_scannet_missing_mask_graph.sh 0,1 c1
#   bash scripts/run_scannet_missing_mask_graph.sh 0,1 c2_observer
#   bash scripts/run_scannet_missing_mask_graph.sh 0,1 c2
#   bash scripts/run_scannet_missing_mask_graph.sh 0,1 c3_observer
#   bash scripts/run_scannet_missing_mask_graph.sh 0,1 b6
#   bash scripts/run_scannet_missing_mask_graph.sh 0,1 b5_b6
#
# Run all 100 validation scenes:
#   BOXFUSION_MASK_GRAPH_FULL100=1 \
#     bash scripts/run_scannet_missing_mask_graph.sh 0,1 supplemental

GPU_SPEC="${1:-0}"
VARIANT="${2:-${BOXFUSION_MASK_GRAPH_VARIANT:-observer}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FULL100="${BOXFUSION_MASK_GRAPH_FULL100:-0}"
ALLOW_RESUME="${BOXFUSION_MASK_GRAPH_ALLOW_RESUME:-0}"
CONFIG="${BOXFUSION_MASK_GRAPH_CONFIG:-$ROOT/config/scannet_online_refinement.yaml}"
YOLOE_CHECKPOINT="${BOXFUSION_MASK_GRAPH_YOLOE_CHECKPOINT:-${BOXFUSION_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}}"
PROPOSAL_PROVIDER="${BOXFUSION_MASK_GRAPH_PROVIDER:-yoloe}"
TEACHER_CACHE_DIRECTORY="${BOXFUSION_MASK_GRAPH_TEACHER_CACHE_DIRECTORY:-}"
TEACHER_CACHE_NAMESPACE="${BOXFUSION_MASK_GRAPH_TEACHER_CACHE_NAMESPACE:-}"
TEACHER_CACHE_MISSING_POLICY="${BOXFUSION_MASK_GRAPH_TEACHER_CACHE_MISSING_POLICY:-error}"
QUALITY_CHECKPOINT="${BOXFUSION_MASK_GRAPH_B6_CHECKPOINT:-${BOXFUSION_QUALITY_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}}"
REFINER_CHECKPOINT="${BOXFUSION_MASK_GRAPH_B5_CHECKPOINT:-${BOXFUSION_REFINER_CHECKPOINT:-$ROOT/models/scannet_b5v2_oriented_refiner_prototype.pt}}"

case "$VARIANT" in
    observer)
        PROFILE="missing_mask_graph_observer"
        NEED_B6=0
        NEED_B5=0
        ;;
    supplemental)
        PROFILE="missing_mask_graph_supplemental"
        NEED_B6=0
        NEED_B5=0
        ;;
    c1)
        PROFILE="missing_mask_graph_c1_recovery"
        NEED_B6=0
        NEED_B5=0
        ;;
    c2_observer)
        PROFILE="missing_mask_graph_c2_geometry_observer"
        NEED_B6=0
        NEED_B5=0
        ;;
    c2)
        PROFILE="missing_mask_graph_c2_geometry"
        NEED_B6=0
        NEED_B5=0
        ;;
    c3_observer)
        PROFILE="missing_mask_graph_c3_stitch_observer"
        NEED_B6=0
        NEED_B5=0
        ;;
    b6)
        PROFILE="missing_mask_graph_b6"
        NEED_B6=1
        NEED_B5=0
        ;;
    b5_b6)
        PROFILE="missing_mask_graph_b5_b6"
        NEED_B6=1
        NEED_B5=1
        ;;
    *)
        echo "Unknown Mask Graph variant: $VARIANT" >&2
        echo "Expected one of: observer, supplemental, c1, c2_observer, c2, c3_observer, b6, b5_b6" >&2
        exit 2
        ;;
esac

if [[ "$FULL100" != "0" && "$FULL100" != "1" ]]; then
    echo "BOXFUSION_MASK_GRAPH_FULL100 must be 0 or 1" >&2
    exit 2
fi
if [[ "$ALLOW_RESUME" != "0" && "$ALLOW_RESUME" != "1" ]]; then
    echo "BOXFUSION_MASK_GRAPH_ALLOW_RESUME must be 0 or 1" >&2
    exit 2
fi

if [[ -n "${BOXFUSION_MASK_GRAPH_SCENE_LIST:-}" ]]; then
    SCENE_LIST="$BOXFUSION_MASK_GRAPH_SCENE_LIST"
    SCOPE_TAG="custom"
elif [[ "$FULL100" == "1" ]]; then
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
    SCOPE_TAG="full100"
else
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
    SCOPE_TAG="ablation10"
fi

if [[ "$VARIANT" == "c1" ]]; then
    DEFAULT_RUN_TAG="maskgraph_c1_recovery_${SCOPE_TAG}_v3"
elif [[ "$VARIANT" == "c2_observer" ]]; then
    DEFAULT_RUN_TAG="maskgraph_c2_geometry_observer_${SCOPE_TAG}_v1"
elif [[ "$VARIANT" == "c2" ]]; then
    DEFAULT_RUN_TAG="maskgraph_c2_geometry_${SCOPE_TAG}_v1"
elif [[ "$VARIANT" == "c3_observer" ]]; then
    DEFAULT_RUN_TAG="maskgraph_c3_stitch_observer_${SCOPE_TAG}_v1"
else
    DEFAULT_RUN_TAG="maskgraph_${VARIANT}_${SCOPE_TAG}_v1"
fi
RUN_TAG="${BOXFUSION_MASK_GRAPH_RUN_TAG:-$DEFAULT_RUN_TAG}"
PRED_ROOT="${BOXFUSION_MASK_GRAPH_PRED_ROOT:-$ROOT/results/$RUN_TAG}"
LOG_ROOT="${BOXFUSION_MASK_GRAPH_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
DIAGNOSTICS_ROOT="${BOXFUSION_MASK_GRAPH_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/$RUN_TAG}"
EVAL_ROOT="${BOXFUSION_MASK_GRAPH_EVAL_ROOT:-$ROOT/evaluation/$RUN_TAG}"

if [[ ! -s "$SCENE_LIST" ]]; then
    echo "Missing or empty Mask Graph scene list: $SCENE_LIST" >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "Missing Mask Graph runtime config: $CONFIG" >&2
    exit 1
fi
if [[ "$PROPOSAL_PROVIDER" != "yoloe" \
      && "$PROPOSAL_PROVIDER" != "cache_only" ]]; then
    echo "BOXFUSION_MASK_GRAPH_PROVIDER must be yoloe or cache_only" >&2
    exit 2
fi
if [[ "$PROPOSAL_PROVIDER" == "yoloe" \
      && ! -f "$YOLOE_CHECKPOINT" ]]; then
    echo "Missing lightweight supplemental-proposal checkpoint: $YOLOE_CHECKPOINT" >&2
    echo "Set BOXFUSION_MASK_GRAPH_YOLOE_CHECKPOINT to a local YOLOE segmentation checkpoint." >&2
    exit 1
fi
if [[ "$PROPOSAL_PROVIDER" == "cache_only" ]]; then
    if [[ -z "$TEACHER_CACHE_DIRECTORY" \
          || ! -d "$TEACHER_CACHE_DIRECTORY" ]]; then
        echo "cache_only requires BOXFUSION_MASK_GRAPH_TEACHER_CACHE_DIRECTORY" >&2
        exit 1
    fi
    if [[ -z "$TEACHER_CACHE_NAMESPACE" ]]; then
        echo "cache_only requires BOXFUSION_MASK_GRAPH_TEACHER_CACHE_NAMESPACE" >&2
        exit 1
    fi
fi
if [[ "$NEED_B6" == "1" && ! -f "$QUALITY_CHECKPOINT" ]]; then
    echo "Variant '$VARIANT' requires the frozen B6 checkpoint: $QUALITY_CHECKPOINT" >&2
    echo "Set BOXFUSION_MASK_GRAPH_B6_CHECKPOINT to scannet_b6_iou_mlp.npz." >&2
    exit 1
fi
if [[ "$NEED_B5" == "1" && ! -f "$REFINER_CHECKPOINT" ]]; then
    echo "Variant '$VARIANT' requires the B5 checkpoint: $REFINER_CHECKPOINT" >&2
    echo "Set BOXFUSION_MASK_GRAPH_B5_CHECKPOINT to the local BoxRefiner checkpoint." >&2
    exit 1
fi

# Do not silently mix predictions produced by a different code/config state.
# The lower-level runner remains resume-friendly after the caller explicitly
# opts in here.
if [[ "$ALLOW_RESUME" != "1" ]]; then
    for directory in "$PRED_ROOT" "$LOG_ROOT" "$DIAGNOSTICS_ROOT" "$EVAL_ROOT"; do
        if [[ -d "$directory" ]] \
            && [[ -n "$(find "$directory" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            echo "Refusing to reuse non-empty Mask Graph experiment directory:" >&2
            echo "  $directory" >&2
            echo "Choose a fresh BOXFUSION_MASK_GRAPH_RUN_TAG." >&2
            echo "For an unchanged interrupted run only, set" >&2
            echo "BOXFUSION_MASK_GRAPH_ALLOW_RESUME=1." >&2
            exit 1
        fi
    done
fi

unset BOXFUSION_DISABLE_ONLINE_REFINEMENT
unset BOXFUSION_JOINT_CHECKPOINT
unset BOXFUSION_JOINT_DETECTOR_BLEND

export BOXFUSION_YOLOE_CHECKPOINT="$YOLOE_CHECKPOINT"
export BOXFUSION_PROPOSAL_PROVIDER="$PROPOSAL_PROVIDER"
if [[ "$PROPOSAL_PROVIDER" == "cache_only" ]]; then
    export BOXFUSION_PROPOSAL_CACHE_DIRECTORY="$TEACHER_CACHE_DIRECTORY"
    export BOXFUSION_PROPOSAL_CACHE_NAMESPACE="$TEACHER_CACHE_NAMESPACE"
    export BOXFUSION_PROPOSAL_CACHE_MISSING_POLICY="$TEACHER_CACHE_MISSING_POLICY"
else
    unset BOXFUSION_PROPOSAL_CACHE_DIRECTORY
    unset BOXFUSION_PROPOSAL_CACHE_NAMESPACE
    unset BOXFUSION_PROPOSAL_CACHE_MISSING_POLICY
fi
export BOXFUSION_ONLINE_CONFIG="$CONFIG"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_ONLINE_ABLATION_PROFILE="$PROFILE"
export BOXFUSION_PROPOSAL_INTERVAL="${BOXFUSION_MASK_GRAPH_PROPOSAL_INTERVAL:-5}"
export BOXFUSION_CANDIDATE_TTL_CLOCK="provider_call"
export BOXFUSION_CANDIDATE_TRACK_TTL="${BOXFUSION_MASK_GRAPH_TRACK_TTL:-4}"
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS="1"
# Do not use the legacy --scannet-min-extent override here: that CLI changes
# both the final ScanNet post-process and the online global-output filter.
# The profile keeps global rows identity (0.0) and applies 0.30 only to new
# graph rows. The final-only override defaults to the B6 anchor's 0.40; the
# C1 runtime keeps that threshold for globals and applies its class-aware
# contract only to C1 supplemental rows.
unset BOXFUSION_SCANNET_MIN_EXTENT
export BOXFUSION_SCANNET_POST_MIN_EXTENT="${BOXFUSION_MASK_GRAPH_POST_MIN_EXTENT:-0.40}"
export BOXFUSION_INFERENCE_SEED="${BOXFUSION_MASK_GRAPH_INFERENCE_SEED:-0}"
export BOXFUSION_EVAL_SEED="${BOXFUSION_MASK_GRAPH_EVAL_SEED:-0}"

export BOXFUSION_ONLINE_PRED_ROOT="$PRED_ROOT"
export BOXFUSION_ONLINE_LOG_ROOT="$LOG_ROOT"
export BOXFUSION_DIAGNOSTICS_ROOT="$DIAGNOSTICS_ROOT"
export BOXFUSION_EVAL_ROOT="$EVAL_ROOT"

# Clear inherited heads first. Each variant below enables only the intended
# branch, which prevents an old shell export from changing the ablation.
unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_QUALITY_CHECKPOINT
unset BOXFUSION_QUALITY_MODE
unset BOXFUSION_QUALITY_DETECTOR_BLEND

if [[ "$NEED_B6" == "1" ]]; then
    export BOXFUSION_QUALITY_CHECKPOINT="$QUALITY_CHECKPOINT"
    export BOXFUSION_QUALITY_MODE="iou_mlp"
    export BOXFUSION_QUALITY_DETECTOR_BLEND="${BOXFUSION_MASK_GRAPH_B6_DETECTOR_BLEND:-0.40}"
fi
if [[ "$NEED_B5" == "1" ]]; then
    export BOXFUSION_REFINER_CHECKPOINT="$REFINER_CHECKPOINT"
fi

scene_count="$(awk 'NF { count += 1 } END { print count + 0 }' "$SCENE_LIST")"
B6_DISPLAY="disabled"
B5_DISPLAY="disabled"
if [[ "$NEED_B6" == "1" ]]; then
    B6_DISPLAY="$QUALITY_CHECKPOINT"
fi
if [[ "$NEED_B5" == "1" ]]; then
    B5_DISPLAY="$REFINER_CHECKPOINT"
fi
echo "Missing-track Mask Graph evaluation"
echo "  variant/profile: $VARIANT / $PROFILE"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  proposal provider: $PROPOSAL_PROVIDER"
if [[ "$PROPOSAL_PROVIDER" == "yoloe" ]]; then
    echo "  proposal checkpoint: $YOLOE_CHECKPOINT"
else
    echo "  teacher cache: $TEACHER_CACHE_DIRECTORY"
    echo "  teacher namespace: $TEACHER_CACHE_NAMESPACE"
fi
echo "  frozen B6: $B6_DISPLAY"
echo "  supplemental-only B5: $B5_DISPLAY"
if [[ "$VARIANT" == "c1" \
      || "$VARIANT" == "c2_observer" \
      || "$VARIANT" == "c2" \
      || "$VARIANT" == "c3_observer" ]]; then
    echo "  C1 gates: absorbed recovery + class-aware extent + Z-aware BEV/planar duplicate + fixed supplemental rank"
    echo "  extent contract: global=$BOXFUSION_SCANNET_POST_MIN_EXTENT; supplemental=class-aware"
    if [[ "$VARIANT" == "c2_observer" ]]; then
        echo "  C2 geometry: observer only (C1 output bit-exact)"
    elif [[ "$VARIANT" == "c2" ]]; then
        echo "  C2 geometry: verified local depth-occupancy refinement"
    elif [[ "$VARIANT" == "c3_observer" ]]; then
        echo "  C2 geometry: verified local depth-occupancy refinement"
        echo "  C3 stitching: cross-lifecycle observer only (C2 output bit-exact)"
    fi
else
    echo "  extent contract: global identity; supplemental=0.30; final ScanNet=$BOXFUSION_SCANNET_POST_MIN_EXTENT"
fi
echo "  tag: $RUN_TAG"
echo "  GPUs: $GPU_SPEC"
if [[ "$NEED_B6" == "1" ]]; then
    echo "  B6 note: checkpoint is frozen; supplemental rows are an explicit source-domain ablation."
fi

exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
