#!/usr/bin/env bash
set -euo pipefail

# Frozen B6 -> unmatched mask -> multi-view residual-track observer.
#
# The default is a two-scene train-only engineering smoke test using the
# immutable SAM3 teacher cache.  No candidate is appended and no B6 box,
# score, count, ID, or ordering may change.
#
#   bash scripts/run_scannet_residual_track_observer.sh 0,1
#
# Select the online YOLOE masks or merge both sources into one graph update:
#
#   BOXFUSION_RESIDUAL_SOURCE_MODE=yoloe bash ... 0,1
#   BOXFUSION_RESIDUAL_SOURCE_MODE=dual  bash ... 0,1
#
# Inspect the fully resolved paths without launching inference:
#
#   BOXFUSION_RESIDUAL_DRY_RUN=1 bash ... 0,1

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SOURCE_MODE="${BOXFUSION_RESIDUAL_SOURCE_MODE:-sam3}"
MIN_SEMANTIC_SCORE="${BOXFUSION_RESIDUAL_MIN_SEMANTIC_SCORE:-0.50}"
DRY_RUN="${BOXFUSION_RESIDUAL_DRY_RUN:-0}"
ALLOW_RESUME="${BOXFUSION_RESIDUAL_ALLOW_RESUME:-0}"
CONFIG="${BOXFUSION_RESIDUAL_CONFIG:-$ROOT/config/scannet_online_refinement.yaml}"
SCENE_LIST="${BOXFUSION_RESIDUAL_SCENE_LIST:-/data/ZhaoX/OVM3D-Dett/boxfusion_p1g_dev/evaluation/data_util/meta_data/scannetv2_train_p1_smoke2.txt}"
FRAMES_ROOT="${BOXFUSION_RESIDUAL_FRAMES_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_b3_dev/data/scannet_train}"
YOLOE_CHECKPOINT="${BOXFUSION_RESIDUAL_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"
QUALITY_CHECKPOINT="${BOXFUSION_RESIDUAL_B6_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
TEACHER_CACHE="${BOXFUSION_RESIDUAL_TEACHER_CACHE:-/data/ZhaoX/OVM3D-Dett/boxfusion_trifusion_dev/cache/sam3_teacher/sam3_teacher_train100_c050_v1}"
TEACHER_NAMESPACE="${BOXFUSION_RESIDUAL_TEACHER_NAMESPACE:-sam3-scannet18-train100-c050-v1}"
CACHE_MISSING_POLICY="${BOXFUSION_RESIDUAL_CACHE_MISSING_POLICY:-error}"
RUN_TAG="${BOXFUSION_RESIDUAL_RUN_TAG:-residual_track_${SOURCE_MODE}_train_smoke2_v1}"
PRED_ROOT="${BOXFUSION_RESIDUAL_PRED_ROOT:-$ROOT/results/residual_track/$RUN_TAG}"
LOG_ROOT="${BOXFUSION_RESIDUAL_LOG_ROOT:-$ROOT/logs/residual_track/$RUN_TAG}"
DIAGNOSTICS_ROOT="${BOXFUSION_RESIDUAL_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/residual_track/$RUN_TAG}"
EVAL_ROOT="${BOXFUSION_RESIDUAL_EVAL_ROOT:-$ROOT/evaluation/residual_track/$RUN_TAG}"

case "$SOURCE_MODE" in
    sam3|yoloe|dual) ;;
    *)
        echo "BOXFUSION_RESIDUAL_SOURCE_MODE must be sam3, yoloe, or dual" >&2
        exit 2
        ;;
esac
if [[ ! "$MIN_SEMANTIC_SCORE" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || ! awk -v value="$MIN_SEMANTIC_SCORE" \
        'BEGIN { exit !(value >= 0.0 && value <= 1.0) }'; then
    echo "BOXFUSION_RESIDUAL_MIN_SEMANTIC_SCORE must lie in [0, 1]" >&2
    exit 2
fi
case "$DRY_RUN" in
    0|1) ;;
    *) echo "BOXFUSION_RESIDUAL_DRY_RUN must be 0 or 1" >&2; exit 2 ;;
esac
case "$ALLOW_RESUME" in
    0|1) ;;
    *) echo "BOXFUSION_RESIDUAL_ALLOW_RESUME must be 0 or 1" >&2; exit 2 ;;
esac
case "$CACHE_MISSING_POLICY" in
    error|empty) ;;
    *)
        echo "BOXFUSION_RESIDUAL_CACHE_MISSING_POLICY must be error or empty" >&2
        exit 2
        ;;
esac

for path in \
    "$CONFIG" \
    "$SCENE_LIST" \
    "$YOLOE_CHECKPOINT" \
    "$QUALITY_CHECKPOINT"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing residual-track input: $path" >&2
        exit 1
    fi
done
for directory in "$FRAMES_ROOT" "$TEACHER_CACHE"; do
    if [[ ! -d "$directory" ]]; then
        echo "Missing residual-track directory: $directory" >&2
        exit 1
    fi
done
if [[ ! -s "$SCENE_LIST" ]]; then
    echo "Residual-track scene list is empty: $SCENE_LIST" >&2
    exit 1
fi
if [[ -z "$TEACHER_NAMESPACE" ]]; then
    echo "BOXFUSION_RESIDUAL_TEACHER_NAMESPACE cannot be empty" >&2
    exit 1
fi

if [[ "$ALLOW_RESUME" != "1" && "$DRY_RUN" != "1" ]]; then
    for directory in \
        "$PRED_ROOT" "$LOG_ROOT" "$DIAGNOSTICS_ROOT" "$EVAL_ROOT"; do
        if [[ -d "$directory" ]] \
            && [[ -n "$(find "$directory" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            echo "Refusing to reuse non-empty residual-track directory:" >&2
            echo "  $directory" >&2
            echo "Choose a fresh BOXFUSION_RESIDUAL_RUN_TAG." >&2
            exit 1
        fi
    done
fi

scene_count="$(awk 'NF { count += 1 } END { print count + 0 }' "$SCENE_LIST")"
echo "Frozen B6 + unmatched-mask residual-track observer"
echo "  profile: residual_track_observer"
echo "  source mode: $SOURCE_MODE"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  frames: $FRAMES_ROOT"
echo "  teacher: cache_only / $TEACHER_CACHE"
echo "  teacher namespace: $TEACHER_NAMESPACE"
echo "  frozen B6: detector blend=0.40 / minimum extent=0.40"
echo "  graph: real depth / provider-call clock / min unique views=2"
echo "  graph minimum semantic score: $MIN_SEMANTIC_SCORE"
echo "  output contract: observer identity; applied candidates=0"
echo "  run tag: $RUN_TAG"
echo "  GPUs: $GPU_SPEC"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete; inference was not launched."
    exit 0
fi

unset BOXFUSION_DISABLE_ONLINE_REFINEMENT
unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_JOINT_CHECKPOINT
unset BOXFUSION_JOINT_DETECTOR_BLEND
unset BOXFUSION_TRIFUSION_GATE_CHECKPOINT
unset BOXFUSION_YIDU_GATE_CHECKPOINT
unset BOXFUSION_PROPOSAL_CACHE_DIRECTORY
unset BOXFUSION_PROPOSAL_CACHE_NAMESPACE
unset BOXFUSION_PROPOSAL_CACHE_MISSING_POLICY

export BOXFUSION_ONLINE_CONFIG="$CONFIG"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_SCANNET_FRAMES_ROOT="$FRAMES_ROOT"
export BOXFUSION_ONLINE_ABLATION_PROFILE="residual_track_observer"
export BOXFUSION_RESIDUAL_TRACK_SOURCE_MODE="$SOURCE_MODE"
export BOXFUSION_RESIDUAL_MIN_SEMANTIC_SCORE="$MIN_SEMANTIC_SCORE"

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

# Detached immutable SAM3 teacher stream.  It remains configured in yoloe
# mode as an identical profile input, but its observations are excluded from
# the residual graph by ``source_mode``.
export BOXFUSION_C4_PROPOSAL_CACHE_DIRECTORY="$TEACHER_CACHE"
export BOXFUSION_C4_PROPOSAL_CACHE_NAMESPACE="$TEACHER_NAMESPACE"
export BOXFUSION_C4_PROPOSAL_CACHE_MISSING_POLICY="$CACHE_MISSING_POLICY"

export BOXFUSION_ONLINE_PRED_ROOT="$PRED_ROOT"
export BOXFUSION_ONLINE_LOG_ROOT="$LOG_ROOT"
export BOXFUSION_DIAGNOSTICS_ROOT="$DIAGNOSTICS_ROOT"
export BOXFUSION_EVAL_ROOT="$EVAL_ROOT"

exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
