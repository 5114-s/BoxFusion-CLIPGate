#!/usr/bin/env bash
set -euo pipefail

# Explicit GPU collection of the audited B5 K=5/P=128 source diagnostics used
# to supervise the SGCDet-inspired head. The B5 observer is prediction-
# preserving and always runs on train scenes; S1 validation uses the separate
# sgcdet_sparse_observer profile.

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/lib_sgcdet_sparse_protocol.sh
source "$SCRIPT_DIR/lib_sgcdet_sparse_protocol.sh"

SCENE_LIST="${BOXFUSION_SGCDET_TRAIN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
VAL_LIST="${BOXFUSION_SGCDET_VAL_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
FRAMES_ROOT="${BOXFUSION_SCANNET_FRAMES_ROOT:-$ROOT/data/scannet_train}"
B6_CHECKPOINT="${BOXFUSION_QUALITY_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
RUN_TAG="${BOXFUSION_SGCDET_COLLECT_TAG:-sgcdet_sparse_observer_train_v1}"

sgcdet_sparse_assert_train_only "$SCENE_LIST" "$VAL_LIST"
sgcdet_sparse_require_directory "$FRAMES_ROOT" "prepared ScanNet train RGB-D frames"
if ! sgcdet_sparse_require_file "$B6_CHECKPOINT" "frozen B6 quality checkpoint"; then
    echo "Set BOXFUSION_QUALITY_CHECKPOINT to a read-only known-good B6 checkpoint." >&2
    exit 1
fi

unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_JOINT_CHECKPOINT
unset BOXFUSION_SGCDET_SPARSE_CHECKPOINT
unset BOXFUSION_DISABLE_ONLINE_REFINEMENT

export BOXFUSION_ONLINE_CONFIG="$ROOT/config/scannet_online_refinement.yaml"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_SCANNET_FRAMES_ROOT="$FRAMES_ROOT"
export BOXFUSION_YOLOE_CHECKPOINT="${BOXFUSION_SGCDET_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"
# The audited B5 dataset join requires its historical strict K=5 observer
# provenance. This profile builds the same K=5/P=128 local tensors while
# preserving every prediction; S1 remains the sparse validation observer.
export BOXFUSION_ONLINE_ABLATION_PROFILE="b5v2_memory_observer"
export BOXFUSION_QUALITY_MODE="iou_mlp"
export BOXFUSION_QUALITY_CHECKPOINT="$B6_CHECKPOINT"
export BOXFUSION_QUALITY_DETECTOR_BLEND="${BOXFUSION_SGCDET_B6_DETECTOR_BLEND:-0.40}"
export BOXFUSION_SCANNET_MIN_EXTENT="${BOXFUSION_SGCDET_MIN_EXTENT:-0.40}"
export BOXFUSION_PROPOSAL_INTERVAL="${BOXFUSION_SGCDET_PROPOSAL_INTERVAL:-5}"
export BOXFUSION_CANDIDATE_TTL_CLOCK="provider_call"
export BOXFUSION_CANDIDATE_TRACK_TTL="3"
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS="0"
export BOXFUSION_INFERENCE_SEED="0"
export BOXFUSION_EVAL_SEED="0"

export BOXFUSION_ONLINE_PRED_ROOT="${BOXFUSION_SGCDET_TRAIN_PREDICTIONS:-$ROOT/results/$RUN_TAG}"
export BOXFUSION_ONLINE_LOG_ROOT="${BOXFUSION_SGCDET_TRAIN_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
export BOXFUSION_DIAGNOSTICS_ROOT="${BOXFUSION_SGCDET_TRAIN_DIAGNOSTICS:-$ROOT/diagnostics/$RUN_TAG}"
export BOXFUSION_EVAL_ROOT="${BOXFUSION_SGCDET_TRAIN_EVAL_ROOT:-$ROOT/evaluation/$RUN_TAG}"

scene_count="$(sgcdet_sparse_scene_count "$SCENE_LIST")"
echo "SGCDet-inspired sparse-refiner train-source observer"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  forbidden validation list: $VAL_LIST"
echo "  frames: $FRAMES_ROOT"
echo "  frozen B6 checkpoint: $B6_CHECKPOINT"
echo "  profile: b5v2_memory_observer (strict K=5 training provenance)"
echo "  diagnostics: $BOXFUSION_DIAGNOSTICS_ROOT"
echo "  GPUs: $GPU_SPEC"

exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
