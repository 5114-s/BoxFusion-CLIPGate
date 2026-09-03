#!/usr/bin/env bash
set -euo pipefail

# Collect output-preserving K=5 diagnostics on ScanNet TRAIN scenes only.
#
# The observer constructs exactly the tensors consumed by the joint local head
# while preserving BoxFusion boxes, scores, ordering, and instance count.  It
# never loads a joint checkpoint and never uses the official validation split
# for supervision.
#
# Usage (run manually after the current GPU experiment has finished):
#   bash scripts/collect_scannet_joint_b356_train.sh 0,1

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENE_LIST="${BOXFUSION_JOINT_B356_TRAIN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
VAL_LIST="${BOXFUSION_JOINT_B356_FORBIDDEN_VAL_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
FRAMES_ROOT="${BOXFUSION_SCANNET_FRAMES_ROOT:-$ROOT/data/scannet_train}"
RUN_TAG="${BOXFUSION_JOINT_B356_COLLECT_TAG:-joint_b356_k5_p128_observer_train_v1}"
MIN_EXTENT="${BOXFUSION_JOINT_B356_MIN_EXTENT:-0.40}"
ALLOW_RESUME="${BOXFUSION_JOINT_B356_ALLOW_COLLECT_RESUME:-0}"

PRED_ROOT="${BOXFUSION_JOINT_B356_TRAIN_PRED_ROOT:-$ROOT/results/$RUN_TAG}"
LOG_ROOT="${BOXFUSION_JOINT_B356_TRAIN_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
DIAGNOSTICS_ROOT="${BOXFUSION_JOINT_B356_TRAIN_DIAGNOSTICS:-$ROOT/diagnostics/$RUN_TAG}"
EVAL_ROOT="${BOXFUSION_JOINT_B356_TRAIN_EVAL_ROOT:-$ROOT/evaluation/$RUN_TAG}"

if [[ ! -s "$SCENE_LIST" ]]; then
    echo "Missing or empty ScanNet train scene list: $SCENE_LIST" >&2
    exit 1
fi
if [[ ! -s "$VAL_LIST" ]]; then
    echo "Missing forbidden ScanNet validation list: $VAL_LIST" >&2
    exit 1
fi
scene_list_name="$(basename -- "$SCENE_LIST")"
scene_list_name_lower="${scene_list_name,,}"
if [[ "$scene_list_name_lower" =~ (^|[_-])(val|validation)([_.-]|$) ]]; then
    echo "Refusing validation-labelled joint training list: $SCENE_LIST" >&2
    exit 1
fi
if [[ ! -d "$FRAMES_ROOT" ]]; then
    echo "Missing prepared ScanNet train RGB-D frames: $FRAMES_ROOT" >&2
    echo "Prepare or link train-only frames before collecting diagnostics." >&2
    exit 1
fi
if [[ "$ALLOW_RESUME" != "0" && "$ALLOW_RESUME" != "1" ]]; then
    echo "BOXFUSION_JOINT_B356_ALLOW_COLLECT_RESUME must be 0 or 1" >&2
    exit 1
fi
if [[ ! "$MIN_EXTENT" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "BOXFUSION_JOINT_B356_MIN_EXTENT must be non-negative" >&2
    exit 1
fi

overlap="$(
    awk '
        NR == FNR { if (NF) forbidden[$1] = 1; next }
        NF && ($1 in forbidden) { print $1; exit }
    ' "$VAL_LIST" "$SCENE_LIST"
)"
if [[ -n "$overlap" ]]; then
    echo "Refusing train/validation leakage; scene appears in val: $overlap" >&2
    exit 1
fi
duplicate="$(
    awk 'NF { count[$1] += 1; if (count[$1] == 2) { print $1; exit } }' \
        "$SCENE_LIST"
)"
if [[ -n "$duplicate" ]]; then
    echo "Refusing duplicate train scene in list: $duplicate" >&2
    exit 1
fi

# Refuse partial prediction/diagnostic pairs, and refuse all reuse unless the
# caller explicitly marks an unchanged interrupted run as resumable.
while IFS= read -r scene || [[ -n "$scene" ]]; do
    [[ -z "$scene" ]] && continue
    prediction="$PRED_ROOT/${scene}_boxes.pkl"
    diagnostic="$DIAGNOSTICS_ROOT/${scene}_tracks.npz"
    if [[ -s "$prediction" && ! -s "$diagnostic" ]]; then
        echo "Incomplete prior joint collection pair for $scene:" >&2
        echo "  prediction exists: $prediction" >&2
        echo "  diagnostic missing: $diagnostic" >&2
        echo "Use a fresh BOXFUSION_JOINT_B356_COLLECT_TAG." >&2
        exit 1
    fi
    if [[ -s "$diagnostic" && ! -s "$prediction" ]]; then
        echo "Incomplete prior joint collection pair for $scene:" >&2
        echo "  diagnostic exists: $diagnostic" >&2
        echo "  prediction missing: $prediction" >&2
        echo "Use a fresh BOXFUSION_JOINT_B356_COLLECT_TAG." >&2
        exit 1
    fi
    if [[ -s "$prediction" && -s "$diagnostic" && "$ALLOW_RESUME" != "1" ]]; then
        echo "Refusing to reuse existing joint train pair for $scene." >&2
        echo "Use a fresh BOXFUSION_JOINT_B356_COLLECT_TAG, or explicitly set" >&2
        echo "BOXFUSION_JOINT_B356_ALLOW_COLLECT_RESUME=1 for an unchanged run." >&2
        exit 1
    fi
done <"$SCENE_LIST"

unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_QUALITY_CHECKPOINT
unset BOXFUSION_QUALITY_MODE
unset BOXFUSION_QUALITY_DETECTOR_BLEND
unset BOXFUSION_JOINT_CHECKPOINT
unset BOXFUSION_JOINT_DETECTOR_BLEND
unset BOXFUSION_DISABLE_ONLINE_REFINEMENT

export BOXFUSION_ONLINE_CONFIG="$ROOT/config/scannet_online_refinement.yaml"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_SCANNET_FRAMES_ROOT="$FRAMES_ROOT"
export BOXFUSION_ONLINE_ABLATION_PROFILE="b5v2_memory_observer"
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
echo "Joint B3/B5/B6-v2 train-only diagnostic collection"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  forbidden validation list: $VAL_LIST"
echo "  profile: b5v2_memory_observer (strict output identity)"
echo "  K/views: 5; points/view: 128"
echo "  minimum extent: $MIN_EXTENT"
echo "  tag: $RUN_TAG"
echo "  predictions: $PRED_ROOT"
echo "  diagnostics: $DIAGNOSTICS_ROOT"
echo "This command collects diagnostics only; it does not train a checkpoint."

exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
