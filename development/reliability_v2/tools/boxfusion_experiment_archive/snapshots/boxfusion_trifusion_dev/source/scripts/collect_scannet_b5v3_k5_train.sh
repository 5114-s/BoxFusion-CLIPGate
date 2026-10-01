#!/usr/bin/env bash
set -euo pipefail

# Collect a train-only, output-preserving K=5 Mask-RGBD memory run.
#
# This is the only supported diagnostics source for the B5-v2 K5 control and
# B5-v3 AP50-aware refiner.  It deliberately uses a new tag and never reads
# the legacy non-K5 b6_quality_observer_train archives.
#
# Usage (after the current GPU experiment has finished):
#   bash scripts/collect_scannet_b5v3_k5_train.sh 0,1

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_TAG="${BOXFUSION_B5V3_K5_COLLECT_TAG:-b5v3_k5_gatealigned_train_extent040_v2}"
SCENE_LIST="${BOXFUSION_B5V3_TRAIN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
VAL_LIST="${BOXFUSION_B5V3_FORBIDDEN_VAL_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
FRAMES_ROOT="${BOXFUSION_SCANNET_FRAMES_ROOT:-$ROOT/data/scannet_train}"
MIN_EXTENT="${BOXFUSION_B5V3_RUNTIME_MIN_EXTENT:-0.40}"
ALLOW_RESUME="${BOXFUSION_B5V3_ALLOW_RESUME:-0}"
PRED_ROOT="${BOXFUSION_B5V3_K5_PRED_ROOT:-$ROOT/results/$RUN_TAG}"
LOG_ROOT="${BOXFUSION_B5V3_K5_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
DIAGNOSTICS_ROOT="${BOXFUSION_B5V3_K5_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/$RUN_TAG}"
EVAL_ROOT="${BOXFUSION_B5V3_K5_EVAL_ROOT:-$ROOT/evaluation/$RUN_TAG}"

if [[ ! -f "$SCENE_LIST" ]]; then
    echo "Missing B5-v3 train scene list: $SCENE_LIST" >&2
    exit 1
fi
if [[ ! -s "$SCENE_LIST" ]]; then
    echo "B5-v3 train scene list is empty: $SCENE_LIST" >&2
    exit 1
fi
if [[ "${SCENE_LIST,,}" == *val* ]]; then
    echo "Refusing validation-labelled K5 collection list: $SCENE_LIST" >&2
    exit 1
fi
if [[ ! -f "$VAL_LIST" ]]; then
    echo "Missing forbidden ScanNet validation list: $VAL_LIST" >&2
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
if [[ ! -d "$FRAMES_ROOT" ]]; then
    echo "Missing prepared ScanNet train frames: $FRAMES_ROOT" >&2
    echo "Prepare or link the train-only RGB-D frames before collection." >&2
    exit 1
fi
if [[ ! "$MIN_EXTENT" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "BOXFUSION_B5V3_RUNTIME_MIN_EXTENT must be non-negative" >&2
    exit 1
fi
if [[ "$ALLOW_RESUME" != "0" && "$ALLOW_RESUME" != "1" ]]; then
    echo "BOXFUSION_B5V3_ALLOW_RESUME must be 0 or 1" >&2
    exit 1
fi

# The driver skips a scene when its prediction exists.  Refuse a half-pair so
# an interrupted run cannot silently produce a mixed/incomplete training set.
while IFS= read -r scene || [[ -n "$scene" ]]; do
    [[ -z "$scene" ]] && continue
    prediction="$PRED_ROOT/${scene}_boxes.pkl"
    diagnostic="$DIAGNOSTICS_ROOT/${scene}_tracks.npz"
    if [[ -s "$prediction" && ! -s "$diagnostic" ]]; then
        echo "Incomplete prior K5 pair for $scene:" >&2
        echo "  prediction exists: $prediction" >&2
        echo "  diagnostic missing: $diagnostic" >&2
        echo "Use a fresh BOXFUSION_B5V3_K5_COLLECT_TAG; do not mix runs." >&2
        exit 1
    elif [[ -s "$diagnostic" && ! -s "$prediction" ]]; then
        echo "Incomplete prior K5 pair for $scene:" >&2
        echo "  diagnostic exists: $diagnostic" >&2
        echo "  prediction missing: $prediction" >&2
        echo "Use a fresh BOXFUSION_B5V3_K5_COLLECT_TAG; do not mix runs." >&2
        exit 1
    elif [[ -s "$prediction" && -s "$diagnostic" && "$ALLOW_RESUME" != "1" ]]; then
        echo "Refusing to reuse an existing complete K5 pair for $scene:" >&2
        echo "  prediction: $prediction" >&2
        echo "  diagnostic: $diagnostic" >&2
        echo "Use a fresh BOXFUSION_B5V3_K5_COLLECT_TAG." >&2
        echo "For an unchanged interrupted run only, explicitly set" >&2
        echo "BOXFUSION_B5V3_ALLOW_RESUME=1." >&2
        exit 1
    fi
done <"$SCENE_LIST"

unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_QUALITY_CHECKPOINT
unset BOXFUSION_QUALITY_MODE
unset BOXFUSION_QUALITY_DETECTOR_BLEND
unset BOXFUSION_DISABLE_ONLINE_REFINEMENT

export BOXFUSION_ONLINE_CONFIG="$ROOT/config/scannet_online_refinement.yaml"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_SCANNET_FRAMES_ROOT="$FRAMES_ROOT"
export BOXFUSION_ONLINE_ABLATION_PROFILE="b5v2_memory_observer"
export BOXFUSION_PROPOSAL_INTERVAL="${BOXFUSION_B5V3_PROPOSAL_INTERVAL:-5}"
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
echo "B5-v3 strict K5 train-memory collection"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  profile: b5v2_memory_observer (output preserving)"
echo "  K: 5; proposal interval: $BOXFUSION_PROPOSAL_INTERVAL"
echo "  runtime minimum extent: $MIN_EXTENT"
echo "  resume existing complete pairs: $ALLOW_RESUME"
echo "  tag: $RUN_TAG"
echo "  predictions: $PRED_ROOT"
echo "  diagnostics: $DIAGNOSTICS_ROOT"
echo "This command collects diagnostics only; it does not train a checkpoint."

exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
