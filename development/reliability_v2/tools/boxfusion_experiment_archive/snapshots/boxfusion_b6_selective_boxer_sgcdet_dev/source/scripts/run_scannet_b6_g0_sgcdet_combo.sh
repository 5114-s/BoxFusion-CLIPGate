#!/usr/bin/env bash
set -euo pipefail

# Frozen Selective-Boxer G0 followed by one SGCDet sparse-refiner ablation.
#
# Fixed10 examples:
#   bash scripts/run_scannet_b6_g0_sgcdet_combo.sh g0 0,1
#   bash scripts/run_scannet_b6_g0_sgcdet_combo.sh observer 0,1
#   bash scripts/run_scannet_b6_g0_sgcdet_combo.sh identity 0,1
#   bash scripts/run_scannet_b6_g0_sgcdet_combo.sh active 0,1
#
# A run with more than ten scenes requires both an explicit scene list and tag.

STAGE="${1:-}"
GPU_SPEC="${2:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/lib_sgcdet_sparse_protocol.sh
source "$SCRIPT_DIR/lib_sgcdet_sparse_protocol.sh"

if [[ -z "$STAGE" ]]; then
    echo "Usage: bash scripts/run_scannet_b6_g0_sgcdet_combo.sh {g0|observer|identity|active} [GPU_SPEC]" >&2
    exit 2
fi

CONFIG="$ROOT/config/scannet_b6_selective_boxer_sgcdet.yaml"
FIXED10_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
SCENE_LIST="${BOXFUSION_COMBO_SCENE_LIST:-$FIXED10_LIST}"
B6_CHECKPOINT="${BOXFUSION_QUALITY_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
YOLOE_CHECKPOINT="${BOXFUSION_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"
IDENTITY_CHECKPOINT="$ROOT/models/scannet_sgcdet_sparse_refiner_identity.pt"
ACTIVE_CHECKPOINT="$ROOT/models/scannet_sgcdet_sparse_refiner.pt"

for dependency in "$CONFIG" "$SCENE_LIST" "$B6_CHECKPOINT" "$YOLOE_CHECKPOINT"; do
    sgcdet_sparse_require_file "$dependency" "combined-ablation dependency"
done

scene_count="$(sgcdet_sparse_scene_count "$SCENE_LIST")"
if [[ "$scene_count" -eq 0 ]]; then
    echo "Scene list is empty: $SCENE_LIST" >&2
    exit 1
fi
if [[ "$scene_count" -gt 10 && -z "${BOXFUSION_COMBO_SCENE_LIST:-}" ]]; then
    echo "A >10-scene run requires BOXFUSION_COMBO_SCENE_LIST." >&2
    exit 1
fi
if [[ "$scene_count" -gt 10 && -z "${BOXFUSION_COMBO_RUN_TAG:-}" ]]; then
    echo "A >10-scene run requires BOXFUSION_COMBO_RUN_TAG." >&2
    exit 1
fi

case "$STAGE" in
    g0)
        PROFILE="quality_only"
        DEFAULT_TAG="g0_frozen_fixed10_v1"
        SPARSE_CHECKPOINT=""
        ;;
    observer)
        PROFILE="sgcdet_sparse_observer"
        DEFAULT_TAG="g0_sgcdet_observer_fixed10_v1"
        SPARSE_CHECKPOINT=""
        ;;
    identity)
        PROFILE="sgcdet_sparse_identity"
        DEFAULT_TAG="g0_sgcdet_identity_fixed10_v1"
        SPARSE_CHECKPOINT="${BOXFUSION_COMBO_SPARSE_CHECKPOINT:-$IDENTITY_CHECKPOINT}"
        sgcdet_sparse_require_file "$SPARSE_CHECKPOINT" "SGCDet identity checkpoint"
        ;;
    active)
        PROFILE="sgcdet_sparse_active"
        DEFAULT_TAG="g0_sgcdet_active_fixed10_v1"
        SPARSE_CHECKPOINT="${BOXFUSION_COMBO_SPARSE_CHECKPOINT:-$ACTIVE_CHECKPOINT}"
        sgcdet_sparse_require_file "$SPARSE_CHECKPOINT" "SGCDet active checkpoint"
        ;;
    *)
        echo "Unknown stage '$STAGE'; expected g0, observer, identity, or active." >&2
        exit 2
        ;;
esac

assert_sha256() {
    local path="$1"
    local expected="$2"
    local description="$3"
    local actual
    actual="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual" != "$expected" ]]; then
        echo "$description SHA256 mismatch: $actual" >&2
        exit 1
    fi
}

assert_sha256 "$B6_CHECKPOINT" \
    "d60abf798edbfa3d7902b42651be7d6053727948f740e05795de6feed60a7071" \
    "Frozen B6 quality checkpoint"
assert_sha256 "$YOLOE_CHECKPOINT" \
    "292bdf157a9ec7315f34b567cb93467c5043cd1889a1cc18abbfdeb88d7a948d" \
    "Frozen YOLOE checkpoint"
if [[ "$STAGE" == "identity" ]]; then
    assert_sha256 "$SPARSE_CHECKPOINT" \
        "0dd38feb4b7d37b3ade1976c3b1db9aeb6256b0ce570c0760876f733e25a10ee" \
        "Frozen SGCDet identity checkpoint"
elif [[ "$STAGE" == "active" ]]; then
    assert_sha256 "$SPARSE_CHECKPOINT" \
        "beda774fc3b8f384b408a14388d6b115704e5039b7a110a187760ac9cfd6d182" \
        "Frozen SGCDet active checkpoint"
fi

duplicate_scene="$(
    awk 'NF && $1 !~ /^#/ {count[$1] += 1} END {
        for (scene in count) if (count[scene] > 1) {print scene; exit}
    }' "$SCENE_LIST"
)"
if [[ -n "$duplicate_scene" ]]; then
    echo "Duplicate scene in scene list: $duplicate_scene" >&2
    exit 1
fi

list_sha="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
list_scope="$(basename "$SCENE_LIST" .txt)-${list_sha:0:12}"
RUN_TAG="${BOXFUSION_COMBO_RUN_TAG:-$DEFAULT_TAG}"
ARTIFACT_SCOPE="$ROOT/results/b6_g0_sgcdet/$RUN_TAG/$list_scope"
LOG_SCOPE="$ROOT/logs/b6_g0_sgcdet/$RUN_TAG/$list_scope"
ONLINE_DIAGNOSTICS="$ROOT/diagnostics/b6_g0_sgcdet/$RUN_TAG/$list_scope/online"
BOXER_DIAGNOSTICS="$ROOT/diagnostics/b6_g0_sgcdet/$RUN_TAG/$list_scope/boxer"
EVAL_SCOPE="$ROOT/evaluation/b6_g0_sgcdet/$RUN_TAG/$list_scope"

# No inherited experimental head may silently enter this ablation.
unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_JOINT_CHECKPOINT
unset BOXFUSION_JOINT_DETECTOR_BLEND
unset BOXFUSION_DISABLE_ONLINE_REFINEMENT
if [[ -n "$SPARSE_CHECKPOINT" ]]; then
    export BOXFUSION_SGCDET_SPARSE_CHECKPOINT="$SPARSE_CHECKPOINT"
else
    unset BOXFUSION_SGCDET_SPARSE_CHECKPOINT
fi

export BOXFUSION_ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
export BOXFUSION_LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
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

# Freeze Selective Boxer G0 for every stage.
export BOXFUSION_BOXER_GATE_MAX_CENTER_SHIFT_M="0.10"
export BOXFUSION_BOXER_GATE_MIN_VOLUME_RATIO="0.50"
export BOXFUSION_BOXER_GATE_MAX_VOLUME_RATIO="2.00"
export BOXFUSION_BOXER_DIAGNOSTICS_ROOT="$BOXER_DIAGNOSTICS"

export BOXFUSION_ONLINE_PRED_ROOT="$ARTIFACT_SCOPE"
export BOXFUSION_ONLINE_LOG_ROOT="$LOG_SCOPE"
export BOXFUSION_DIAGNOSTICS_ROOT="$ONLINE_DIAGNOSTICS"
export BOXFUSION_EVAL_ROOT="$EVAL_SCOPE"

echo "B6 + Selective Boxer G0 + SGCDet sparse-refiner ablation"
echo "  stage/profile: $STAGE/$PROFILE"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  scene-list SHA256: $list_sha"
echo "  score/min-extent: 0.40/0.40"
echo "  G0 gate: center<=0.10 m, volume=[0.50,2.00]"
echo "  B6 quality: iou_mlp, detector blend=0.40"
echo "  sparse checkpoint: ${SPARSE_CHECKPOINT:-disabled}"
echo "  predictions: $ARTIFACT_SCOPE"
echo "  online diagnostics: $ONLINE_DIAGNOSTICS"
echo "  Boxer diagnostics: $BOXER_DIAGNOSTICS"
echo "  GPUs: $GPU_SPEC"

exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
