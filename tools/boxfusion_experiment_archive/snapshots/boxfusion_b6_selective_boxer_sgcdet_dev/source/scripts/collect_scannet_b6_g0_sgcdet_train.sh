#!/usr/bin/env bash
set -euo pipefail

# Collect train-only, prediction-preserving diagnostics in the exact frozen
# B6 + Selective-Boxer G0 distribution.  This never evaluates train scenes and
# never reads/writes the validation CuTR replay cache.

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/lib_sgcdet_sparse_protocol.sh
source "$SCRIPT_DIR/lib_sgcdet_sparse_protocol.sh"

CONFIG="$ROOT/config/scannet_b6_selective_boxer_sgcdet.yaml"
SCENE_LIST="${BOXFUSION_G0_SGCDET_TRAIN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
VAL_LIST="${BOXFUSION_G0_SGCDET_FORBIDDEN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
FRAMES_ROOT="${BOXFUSION_SCANNET_FRAMES_ROOT:-$ROOT/data/scannet_train}"
B6_CHECKPOINT="${BOXFUSION_QUALITY_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
YOLOE_CHECKPOINT="${BOXFUSION_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"
RUN_TAG="${BOXFUSION_G0_SGCDET_COLLECT_TAG:-g0_sgcdet_sparse_observer_train_v1}"

sgcdet_sparse_assert_train_only "$SCENE_LIST" "$VAL_LIST"
sgcdet_sparse_require_directory "$FRAMES_ROOT" "prepared ScanNet train RGB-D frames"
for dependency in "$CONFIG" "$B6_CHECKPOINT" "$YOLOE_CHECKPOINT"; do
    sgcdet_sparse_require_file "$dependency" "G0 train-collection dependency"
done

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
assert_sha256 "$CONFIG" \
    "54c4e7686edfc0ecd7bbe1e21e7fba79063e6ea52dedf39ba1dbc95a127d6b36" \
    "Frozen combined config"
assert_sha256 "$B6_CHECKPOINT" \
    "d60abf798edbfa3d7902b42651be7d6053727948f740e05795de6feed60a7071" \
    "Frozen B6 checkpoint"
assert_sha256 "$YOLOE_CHECKPOINT" \
    "292bdf157a9ec7315f34b567cb93467c5043cd1889a1cc18abbfdeb88d7a948d" \
    "Frozen YOLOE checkpoint"

list_sha="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
list_scope="$(basename "$SCENE_LIST" .txt)-${list_sha:0:12}"
PREDICTION_ROOT="${BOXFUSION_G0_SGCDET_TRAIN_PREDICTIONS:-$ROOT/results/b6_g0_sgcdet_train/$RUN_TAG/$list_scope}"
LOG_ROOT="${BOXFUSION_G0_SGCDET_TRAIN_LOG_ROOT:-$ROOT/logs/b6_g0_sgcdet_train/$RUN_TAG/$list_scope}"
DIAGNOSTICS_ROOT="${BOXFUSION_G0_SGCDET_TRAIN_DIAGNOSTICS:-$ROOT/diagnostics/b6_g0_sgcdet_train/$RUN_TAG/$list_scope/online}"
BOXER_ROOT="${BOXFUSION_G0_SGCDET_TRAIN_BOXER_DIAGNOSTICS:-$ROOT/diagnostics/b6_g0_sgcdet_train/$RUN_TAG/$list_scope/boxer}"
SUPPLEMENTAL_CACHE="$ROOT/cache/b6_g0_sgcdet_train/$RUN_TAG/$list_scope/yoloe"
MANIFEST="${BOXFUSION_G0_SGCDET_TRAIN_MANIFEST:-$ROOT/manifests/b6_g0_sgcdet_train/$RUN_TAG/$list_scope/collection_manifest.json}"

# No inherited learned geometry head may enter observer collection.
unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_JOINT_CHECKPOINT
unset BOXFUSION_JOINT_DETECTOR_BLEND
unset BOXFUSION_SGCDET_SPARSE_CHECKPOINT
unset BOXFUSION_DISABLE_ONLINE_REFINEMENT

export BOXFUSION_ONLINE_CONFIG="$CONFIG"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_SCANNET_FRAMES_ROOT="$FRAMES_ROOT"
export BOXFUSION_ONLINE_ABLATION_PROFILE="sgcdet_sparse_observer"
export BOXFUSION_QUALITY_MODE="iou_mlp"
export BOXFUSION_QUALITY_CHECKPOINT="$B6_CHECKPOINT"
export BOXFUSION_QUALITY_DETECTOR_BLEND="0.40"
export BOXFUSION_YOLOE_CHECKPOINT="$YOLOE_CHECKPOINT"
export BOXFUSION_SCANNET_MIN_EXTENT="0.40"
export BOXFUSION_PROPOSAL_INTERVAL="5"
export BOXFUSION_CANDIDATE_TTL_CLOCK="provider_call"
export BOXFUSION_CANDIDATE_TRACK_TTL="3"
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS="0"
export BOXFUSION_INFERENCE_SEED="0"
export BOXFUSION_EVAL_SEED="0"

# Train scenes do not exist in the immutable validation replay cache.  Run
# fresh CuTR and keep all caches/artifacts in a train-only namespace.
export BOXFUSION_PROPOSAL_CACHE_MODE_OVERRIDE="disabled"
export BOXFUSION_SUPPLEMENTAL_CACHE_DIRECTORY="$SUPPLEMENTAL_CACHE"
export BOXFUSION_SKIP_EVALUATION="1"

export BOXFUSION_BOXER_GATE_MAX_CENTER_SHIFT_M="0.10"
export BOXFUSION_BOXER_GATE_MIN_VOLUME_RATIO="0.50"
export BOXFUSION_BOXER_GATE_MAX_VOLUME_RATIO="2.00"
export BOXFUSION_BOXER_DIAGNOSTICS_ROOT="$BOXER_ROOT"
export BOXFUSION_ONLINE_PRED_ROOT="$PREDICTION_ROOT"
export BOXFUSION_ONLINE_LOG_ROOT="$LOG_ROOT"
export BOXFUSION_DIAGNOSTICS_ROOT="$DIAGNOSTICS_ROOT"
export BOXFUSION_EVAL_ROOT="$ROOT/evaluation/b6_g0_sgcdet_train/$RUN_TAG/$list_scope"

echo "G0+B6 same-distribution SGCDet train-only collection"
echo "  scenes: $(sgcdet_sparse_scene_count "$SCENE_LIST") from $SCENE_LIST"
echo "  forbidden validation list: $VAL_LIST"
echo "  profile: sgcdet_sparse_observer (strict output identity)"
echo "  CuTR proposal cache: disabled (fresh train inference)"
echo "  G0 gate: center<=0.10 m, volume=[0.50,2.00]"
echo "  predictions: $PREDICTION_ROOT"
echo "  diagnostics: $DIAGNOSTICS_ROOT"
echo "  Boxer diagnostics: $BOXER_ROOT"
echo "  manifest: $MANIFEST"
echo "  GPUs: $GPU_SPEC"

bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"

mkdir -p "$(dirname "$MANIFEST")"
AUDIT_MODE=()
if [[ -e "$MANIFEST" ]]; then
    AUDIT_MODE+=(--verify-existing)
fi
"${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}/bin/python" \
    "$ROOT/tools/audit_g0_sgcdet_train_collection.py" \
    --scene-list "$SCENE_LIST" \
    --forbidden-scene-list "$VAL_LIST" \
    --prediction-root "$PREDICTION_ROOT" \
    --diagnostics-root "$DIAGNOSTICS_ROOT" \
    --boxer-diagnostics-root "$BOXER_ROOT" \
    --log-root "$LOG_ROOT" \
    --config "$CONFIG" \
    --b6-checkpoint "$B6_CHECKPOINT" \
    --yoloe-checkpoint "$YOLOE_CHECKPOINT" \
    --output "$MANIFEST" \
    "${AUDIT_MODE[@]}"

echo "Audited G0+B6 train collection manifest: $MANIFEST"
