#!/usr/bin/env bash
set -euo pipefail

# Validation protocol for the G0-distribution retrained sparse head.
# Usage: bash scripts/run_scannet_b6_g0_sgcdet_retrained.sh {g0|observer|identity|active} 0,1

STAGE="${1:-}"
GPU_SPEC="${2:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/lib_sgcdet_sparse_protocol.sh
source "$SCRIPT_DIR/lib_sgcdet_sparse_protocol.sh"

if [[ -z "$STAGE" ]]; then
    echo "Usage: bash scripts/run_scannet_b6_g0_sgcdet_retrained.sh {g0|observer|identity|active} [GPU_SPEC]" >&2
    exit 2
fi

CONFIG="$ROOT/config/scannet_b6_selective_boxer_sgcdet.yaml"
FIXED10_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
SCENE_LIST="${BOXFUSION_RETRAINED_SCENE_LIST:-$FIXED10_LIST}"
B6_CHECKPOINT="$ROOT/models/scannet_b6_iou_mlp.npz"
YOLOE_CHECKPOINT="$ROOT/models/yoloe-11s-seg-pf.pt"
ACTIVE_CHECKPOINT="${BOXFUSION_G0_SGCDET_ACTIVE_CHECKPOINT:-$ROOT/models/scannet_b6_g0_sgcdet_sparse_refiner_v1.pt}"
IDENTITY_CHECKPOINT="${BOXFUSION_G0_SGCDET_IDENTITY_CHECKPOINT:-$ROOT/models/scannet_b6_g0_sgcdet_sparse_refiner_identity_v1.pt}"

TRAIN_LIST="${BOXFUSION_G0_SGCDET_TRAIN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
TRAIN_TAG="${BOXFUSION_G0_SGCDET_COLLECT_TAG:-g0_sgcdet_sparse_observer_train_v1}"
train_sha="$(sha256sum "$TRAIN_LIST" | awk '{print $1}')"
train_scope="$(basename "$TRAIN_LIST" .txt)-${train_sha:0:12}"
MANIFEST="${BOXFUSION_G0_SGCDET_TRAIN_MANIFEST:-$ROOT/manifests/b6_g0_sgcdet_train/$TRAIN_TAG/$train_scope/collection_manifest.json}"

case "$STAGE" in
    g0)
        PROFILE="quality_only"
        DEFAULT_TAG="g0_retrained_frozen_fixed10_v1"
        SPARSE_CHECKPOINT=""
        ;;
    observer)
        PROFILE="sgcdet_sparse_observer"
        DEFAULT_TAG="g0_retrained_observer_fixed10_v1"
        SPARSE_CHECKPOINT=""
        ;;
    identity)
        PROFILE="sgcdet_sparse_identity"
        DEFAULT_TAG="g0_retrained_identity_fixed10_v1"
        SPARSE_CHECKPOINT="$IDENTITY_CHECKPOINT"
        ;;
    active)
        PROFILE="sgcdet_sparse_active"
        DEFAULT_TAG="g0_retrained_active_fixed10_v1"
        SPARSE_CHECKPOINT="$ACTIVE_CHECKPOINT"
        ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        exit 2
        ;;
esac

for path in "$CONFIG" "$SCENE_LIST" "$B6_CHECKPOINT" "$YOLOE_CHECKPOINT"; do
    sgcdet_sparse_require_file "$path" "retrained-ablation dependency"
done
if [[ -n "$SPARSE_CHECKPOINT" ]]; then
    for path in "$ACTIVE_CHECKPOINT" "$IDENTITY_CHECKPOINT" "$MANIFEST"; do
        sgcdet_sparse_require_file "$path" "paired retrained checkpoint dependency"
    done
    "${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}/bin/python" \
        "$ROOT/tools/audit_g0_sgcdet_retrained_checkpoints.py" \
        --active "$ACTIVE_CHECKPOINT" \
        --identity "$IDENTITY_CHECKPOINT" \
        --manifest "$MANIFEST"
fi

scene_count="$(sgcdet_sparse_scene_count "$SCENE_LIST")"
if [[ "$scene_count" -gt 10 && -z "${BOXFUSION_RETRAINED_SCENE_LIST:-}" ]]; then
    echo "A >10-scene run requires BOXFUSION_RETRAINED_SCENE_LIST." >&2
    exit 1
fi
if [[ "$scene_count" -gt 10 && -z "${BOXFUSION_RETRAINED_RUN_TAG:-}" ]]; then
    echo "A >10-scene run requires BOXFUSION_RETRAINED_RUN_TAG." >&2
    exit 1
fi

assert_sha256() {
    local path="$1" expected="$2" label="$3" actual
    actual="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual" != "$expected" ]]; then
        echo "$label SHA256 mismatch: $actual" >&2
        exit 1
    fi
}
assert_sha256 "$B6_CHECKPOINT" \
    "d60abf798edbfa3d7902b42651be7d6053727948f740e05795de6feed60a7071" \
    "Frozen B6 checkpoint"
assert_sha256 "$YOLOE_CHECKPOINT" \
    "292bdf157a9ec7315f34b567cb93467c5043cd1889a1cc18abbfdeb88d7a948d" \
    "Frozen YOLOE checkpoint"

list_sha="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
list_scope="$(basename "$SCENE_LIST" .txt)-${list_sha:0:12}"
RUN_TAG="${BOXFUSION_RETRAINED_RUN_TAG:-$DEFAULT_TAG}"
PREDICTION_ROOT="$ROOT/results/b6_g0_sgcdet/$RUN_TAG/$list_scope"
LOG_ROOT="$ROOT/logs/b6_g0_sgcdet/$RUN_TAG/$list_scope"
ONLINE_DIAGNOSTICS="$ROOT/diagnostics/b6_g0_sgcdet/$RUN_TAG/$list_scope/online"
BOXER_DIAGNOSTICS="$ROOT/diagnostics/b6_g0_sgcdet/$RUN_TAG/$list_scope/boxer"
EVAL_ROOT="$ROOT/evaluation/b6_g0_sgcdet/$RUN_TAG/$list_scope"

unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_JOINT_CHECKPOINT
unset BOXFUSION_JOINT_DETECTOR_BLEND
unset BOXFUSION_DISABLE_ONLINE_REFINEMENT
unset BOXFUSION_PROPOSAL_CACHE_MODE_OVERRIDE
unset BOXFUSION_SUPPLEMENTAL_CACHE_DIRECTORY
unset BOXFUSION_SKIP_EVALUATION
if [[ -n "$SPARSE_CHECKPOINT" ]]; then
    export BOXFUSION_SGCDET_SPARSE_CHECKPOINT="$SPARSE_CHECKPOINT"
else
    unset BOXFUSION_SGCDET_SPARSE_CHECKPOINT
fi

export BOXFUSION_ONLINE_CONFIG="$CONFIG"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_ONLINE_ABLATION_PROFILE="$PROFILE"
export BOXFUSION_QUALITY_MODE="iou_mlp"
export BOXFUSION_QUALITY_CHECKPOINT="$B6_CHECKPOINT"
export BOXFUSION_QUALITY_DETECTOR_BLEND="0.40"
export BOXFUSION_SCANNET_MIN_EXTENT="0.40"
export BOXFUSION_PROPOSAL_INTERVAL="5"
export BOXFUSION_CANDIDATE_TTL_CLOCK="provider_call"
export BOXFUSION_CANDIDATE_TRACK_TTL="3"
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS="0"
export BOXFUSION_INFERENCE_SEED="0"
export BOXFUSION_EVAL_SEED="0"
export BOXFUSION_YOLOE_CHECKPOINT="$YOLOE_CHECKPOINT"
export BOXFUSION_BOXER_GATE_MAX_CENTER_SHIFT_M="0.10"
export BOXFUSION_BOXER_GATE_MIN_VOLUME_RATIO="0.50"
export BOXFUSION_BOXER_GATE_MAX_VOLUME_RATIO="2.00"
export BOXFUSION_BOXER_DIAGNOSTICS_ROOT="$BOXER_DIAGNOSTICS"
export BOXFUSION_ONLINE_PRED_ROOT="$PREDICTION_ROOT"
export BOXFUSION_ONLINE_LOG_ROOT="$LOG_ROOT"
export BOXFUSION_DIAGNOSTICS_ROOT="$ONLINE_DIAGNOSTICS"
export BOXFUSION_EVAL_ROOT="$EVAL_ROOT"

echo "G0-distribution retrained SGCDet ablation"
echo "  stage/profile: $STAGE/$PROFILE"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  score/min-extent: 0.40/0.40"
echo "  G0 gate: center<=0.10 m, volume=[0.50,2.00]"
echo "  sparse checkpoint: ${SPARSE_CHECKPOINT:-disabled}"
if [[ -n "$SPARSE_CHECKPOINT" ]]; then
    echo "  sparse checkpoint SHA256: $(sha256sum "$SPARSE_CHECKPOINT" | awk '{print $1}')"
fi
echo "  predictions: $PREDICTION_ROOT"
echo "  GPUs: $GPU_SPEC"

exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
