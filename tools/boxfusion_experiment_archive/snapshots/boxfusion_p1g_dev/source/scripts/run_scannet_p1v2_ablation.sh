#!/usr/bin/env bash
set -euo pipefail

# Isolated P1-v2 residual-proposal ablation.
#
# P1R changes only the train-only target assignment:
#   per_voxel_mlp + snapshot_inside_only
# P1S changes only the P1R proposal head:
#   native_sparse_context_v1 + snapshot_inside_only
#
# Both stages are diagnostics-only and explicitly disable every P2 branch.
#
# Usage:
#   BOXFUSION_P1V2_CHECKPOINT="$PWD/models/scannet_p1r_snapshot_inside.pt" \
#     bash scripts/run_scannet_p1v2_ablation.sh P1R 0,1
#
#   BOXFUSION_P1V2_CHECKPOINT="$PWD/models/scannet_p1s_native_sparse.pt" \
#     bash scripts/run_scannet_p1v2_ablation.sh P1S 0,1

STAGE="${1:-}"
GPU_SPEC="${2:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

case "${STAGE^^}" in
    P1R)
        STAGE=P1R
        PROFILE=p1r_snapshot_target_residual_observer
        DEFAULT_CHECKPOINT="$ROOT/models/scannet_p1r_snapshot_inside.pt"
        ;;
    P1S)
        STAGE=P1S
        PROFILE=p1s_native_sparse_context_observer
        DEFAULT_CHECKPOINT="$ROOT/models/scannet_p1s_native_sparse.pt"
        ;;
    *)
        echo "Stage must be P1R or P1S" >&2
        exit 2
        ;;
esac

FULL100="${BOXFUSION_P1V2_FULL100:-0}"
case "$FULL100" in
    0|1) ;;
    *)
        echo "BOXFUSION_P1V2_FULL100 must be 0 or 1" >&2
        exit 2
        ;;
esac
if [[ -n "${BOXFUSION_P1V2_SCENE_LIST:-}" ]]; then
    SCENE_LIST="$BOXFUSION_P1V2_SCENE_LIST"
    SCOPE=custom
elif [[ "$FULL100" == "1" ]]; then
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
    SCOPE=full100
else
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
    SCOPE=ablation10
fi

CHECKPOINT="${BOXFUSION_P1V2_CHECKPOINT:-$DEFAULT_CHECKPOINT}"
QUALITY_CHECKPOINT="${BOXFUSION_P_B6_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
YOLOE_CHECKPOINT="${BOXFUSION_P_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"
CONFIG="${BOXFUSION_P_CONFIG:-$ROOT/config/scannet_online_refinement.yaml}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"

for path in \
    "$PYTHON" "$SCENE_LIST" "$CHECKPOINT" "$QUALITY_CHECKPOINT" \
    "$YOLOE_CHECKPOINT" "$CONFIG"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing P1-v2 input: $path" >&2
        exit 1
    fi
done

RUN_TAG="${BOXFUSION_P1V2_RUN_TAG:-${STAGE,,}_${SCOPE}_b6frozen_v1}"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "BOXFUSION_P1V2_RUN_TAG contains unsafe path characters" >&2
    exit 2
fi

# No downstream P module is permitted in this ablation.
unset BOXFUSION_P2_OCCUPANCY_CHECKPOINT
unset BOXFUSION_REFINER_CHECKPOINT BOXFUSION_JOINT_CHECKPOINT
unset BOXFUSION_TRIFUSION_GATE_CHECKPOINT BOXFUSION_YIDU_GATE_CHECKPOINT
export BOXFUSION_ENV_ROOT="$ENV_ROOT"
export BOXFUSION_ONLINE_CONFIG="$CONFIG"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_ONLINE_ABLATION_PROFILE="$PROFILE"
export BOXFUSION_P_STAGE="$STAGE"
export BOXFUSION_P1_RESIDUAL_MODE=infer
export BOXFUSION_P1_COLLECT_VOXELS=0
export BOXFUSION_P1_RESIDUAL_CHECKPOINT="$CHECKPOINT"
export BOXFUSION_PROPOSAL_PROVIDER=yoloe
export BOXFUSION_YOLOE_CHECKPOINT="$YOLOE_CHECKPOINT"
export BOXFUSION_QUALITY_MODE=iou_mlp
export BOXFUSION_QUALITY_CHECKPOINT="$QUALITY_CHECKPOINT"
export BOXFUSION_QUALITY_DETECTOR_BLEND="${BOXFUSION_P_B6_DETECTOR_BLEND:-0.40}"
export BOXFUSION_SCANNET_MIN_EXTENT="${BOXFUSION_P_B6_MIN_EXTENT:-0.40}"
export BOXFUSION_PROPOSAL_INTERVAL="${BOXFUSION_P_PROPOSAL_INTERVAL:-5}"
export BOXFUSION_CANDIDATE_TTL_CLOCK=provider_call
export BOXFUSION_CANDIDATE_TRACK_TTL=3
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS=0
export BOXFUSION_INFERENCE_SEED=0
export BOXFUSION_EVAL_SEED=0
export BOXFUSION_SKIP_EVALUATION=0
export BOXFUSION_ONLINE_PRED_ROOT="$ROOT/results/p1v2_ablation/$RUN_TAG"
export BOXFUSION_ONLINE_LOG_ROOT="$ROOT/logs/p1v2_ablation/$RUN_TAG"
export BOXFUSION_DIAGNOSTICS_ROOT="$ROOT/diagnostics/p1v2_ablation/$RUN_TAG"
export BOXFUSION_EVAL_ROOT="$ROOT/evaluation/p1v2_ablation/$RUN_TAG"
export BOXFUSION_P_MANIFEST="$ROOT/logs/p1v2_ablation/$RUN_TAG/run_manifest.json"

echo "P1-v2 ablation: stage=$STAGE, scope=$SCOPE, tag=$RUN_TAG, GPUs=$GPU_SPEC"
exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
