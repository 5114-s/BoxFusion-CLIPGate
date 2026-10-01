#!/usr/bin/env bash
set -euo pipefail

# Incremental fixed10 protocol for frozen B6 + one local sparse-refiner module.
#
# Usage:
#   bash scripts/run_scannet_b6_sgcdet_sparse_refiner.sh s0 0,1
#   bash scripts/run_scannet_b6_sgcdet_sparse_refiner.sh s1 0,1
#   bash scripts/run_scannet_b6_sgcdet_sparse_refiner.sh s2 0,1
#   bash scripts/run_scannet_b6_sgcdet_sparse_refiner.sh s3 0,1

STAGE="${1:-}"
GPU_SPEC="${2:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/lib_sgcdet_sparse_protocol.sh
source "$SCRIPT_DIR/lib_sgcdet_sparse_protocol.sh"

if [[ -z "$STAGE" ]]; then
    echo "Usage: bash scripts/run_scannet_b6_sgcdet_sparse_refiner.sh {s0|s1|s2|s3} [GPU_SPEC]" >&2
    exit 2
fi

FIXED10_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
SCENE_LIST="${BOXFUSION_SGCDET_SCENE_LIST:-$FIXED10_LIST}"
B6_CHECKPOINT="${BOXFUSION_QUALITY_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
IDENTITY_CHECKPOINT="$ROOT/models/scannet_sgcdet_sparse_refiner_identity.pt"
ACTIVE_CHECKPOINT="$ROOT/models/scannet_sgcdet_sparse_refiner.pt"

sgcdet_sparse_require_file "$SCENE_LIST" "ScanNet evaluation scene list"
if ! sgcdet_sparse_require_file "$B6_CHECKPOINT" "frozen B6 quality checkpoint"; then
    echo "Set BOXFUSION_QUALITY_CHECKPOINT to a read-only known-good B6 checkpoint." >&2
    exit 1
fi
scene_count="$(sgcdet_sparse_scene_count "$SCENE_LIST")"
if [[ "$scene_count" -eq 0 ]]; then
    echo "ScanNet evaluation scene list is empty: $SCENE_LIST" >&2
    exit 1
fi
if [[ "$scene_count" -gt 10 && -z "${BOXFUSION_SGCDET_SCENE_LIST:-}" ]]; then
    echo "A >10-scene run requires an explicit BOXFUSION_SGCDET_SCENE_LIST." >&2
    echo "The implicit default is fixed10; full100 must never start accidentally." >&2
    exit 1
fi
if [[ "$scene_count" -gt 10 && -z "${BOXFUSION_SGCDET_RUN_TAG:-}" ]]; then
    echo "A >10-scene run also requires an explicit BOXFUSION_SGCDET_RUN_TAG." >&2
    echo "This prevents full100 from reusing a fixed10 output directory." >&2
    exit 1
fi

case "$STAGE" in
    s0)
        PROFILE="quality_only"
        DEFAULT_TAG="sgcdet_sparse_s0_frozen_b6_fixed10_v1"
        SPARSE_CHECKPOINT=""
        ;;
    s1)
        PROFILE="sgcdet_sparse_observer"
        DEFAULT_TAG="sgcdet_sparse_s1_observer_fixed10_v1"
        SPARSE_CHECKPOINT=""
        ;;
    s2)
        PROFILE="sgcdet_sparse_identity"
        DEFAULT_TAG="sgcdet_sparse_s2_identity_fixed10_v1"
        SPARSE_CHECKPOINT="${BOXFUSION_SGCDET_SPARSE_CHECKPOINT:-$IDENTITY_CHECKPOINT}"
        sgcdet_sparse_require_file "$SPARSE_CHECKPOINT" "SGCDet-inspired sparse identity checkpoint"
        ;;
    s3)
        PROFILE="sgcdet_sparse_active"
        DEFAULT_TAG="sgcdet_sparse_s3_active_fixed10_v1"
        SPARSE_CHECKPOINT="${BOXFUSION_SGCDET_SPARSE_CHECKPOINT:-$ACTIVE_CHECKPOINT}"
        sgcdet_sparse_require_file "$SPARSE_CHECKPOINT" "trained SGCDet-inspired sparse-refiner checkpoint"
        ;;
    *)
        echo "Unknown stage '$STAGE'; expected s0, s1, s2, or s3." >&2
        exit 2
        ;;
esac

RUN_TAG="${BOXFUSION_SGCDET_RUN_TAG:-$DEFAULT_TAG}"

unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_JOINT_CHECKPOINT
unset BOXFUSION_DISABLE_ONLINE_REFINEMENT
if [[ -n "$SPARSE_CHECKPOINT" ]]; then
    export BOXFUSION_SGCDET_SPARSE_CHECKPOINT="$SPARSE_CHECKPOINT"
else
    unset BOXFUSION_SGCDET_SPARSE_CHECKPOINT
fi

export BOXFUSION_ONLINE_CONFIG="$ROOT/config/scannet_online_refinement.yaml"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_YOLOE_CHECKPOINT="${BOXFUSION_SGCDET_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"
export BOXFUSION_ONLINE_ABLATION_PROFILE="$PROFILE"
export BOXFUSION_QUALITY_MODE="iou_mlp"
export BOXFUSION_QUALITY_CHECKPOINT="$B6_CHECKPOINT"
# Freeze the exact historical best-B6 protocol.  Changing either default
# would confound the sparse-refiner ablation with a score/filter ablation.
export BOXFUSION_QUALITY_DETECTOR_BLEND="${BOXFUSION_SGCDET_B6_DETECTOR_BLEND:-0.40}"
export BOXFUSION_SCANNET_MIN_EXTENT="${BOXFUSION_SGCDET_MIN_EXTENT:-0.40}"
export BOXFUSION_PROPOSAL_INTERVAL="${BOXFUSION_SGCDET_PROPOSAL_INTERVAL:-5}"
export BOXFUSION_CANDIDATE_TTL_CLOCK="provider_call"
export BOXFUSION_CANDIDATE_TRACK_TTL="3"
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS="0"
export BOXFUSION_INFERENCE_SEED="0"
export BOXFUSION_EVAL_SEED="0"

export BOXFUSION_ONLINE_PRED_ROOT="${BOXFUSION_SGCDET_PRED_ROOT:-$ROOT/results/$RUN_TAG}"
export BOXFUSION_ONLINE_LOG_ROOT="${BOXFUSION_SGCDET_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
export BOXFUSION_DIAGNOSTICS_ROOT="${BOXFUSION_SGCDET_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/$RUN_TAG}"
export BOXFUSION_EVAL_ROOT="${BOXFUSION_SGCDET_EVAL_ROOT:-$ROOT/evaluation/$RUN_TAG}"

echo "Frozen B6 + SGCDet-inspired local sparse refiner"
echo "  stage/profile: ${STAGE}/${PROFILE}"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  frozen B6 checkpoint: $B6_CHECKPOINT"
echo "  sparse checkpoint: ${SPARSE_CHECKPOINT:-disabled}"
echo "  run tag: $RUN_TAG"
echo "  prediction root: $BOXFUSION_ONLINE_PRED_ROOT"
echo "  GPUs: $GPU_SPEC"

exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
