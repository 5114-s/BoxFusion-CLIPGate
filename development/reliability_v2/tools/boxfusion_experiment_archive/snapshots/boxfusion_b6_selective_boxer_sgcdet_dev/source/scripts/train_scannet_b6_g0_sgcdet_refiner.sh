#!/usr/bin/env bash
set -euo pipefail

# Build and train the G0-distribution sparse refiner on CPU.  The active
# checkpoint is published only if the source data has AP50 crossing potential
# and at least one held-out-train crossing survives the learned candidate.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/lib_sgcdet_sparse_protocol.sh
source "$SCRIPT_DIR/lib_sgcdet_sparse_protocol.sh"

ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

SCENE_LIST="${BOXFUSION_G0_SGCDET_TRAIN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
VAL_LIST="${BOXFUSION_G0_SGCDET_FORBIDDEN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
RUN_TAG="${BOXFUSION_G0_SGCDET_COLLECT_TAG:-g0_sgcdet_sparse_observer_train_v1}"
list_sha="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
list_scope="$(basename "$SCENE_LIST" .txt)-${list_sha:0:12}"

CONFIG="$ROOT/config/scannet_b6_selective_boxer_sgcdet.yaml"
B6_CHECKPOINT="${BOXFUSION_QUALITY_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
YOLOE_CHECKPOINT="${BOXFUSION_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"
PREDICTION_ROOT="${BOXFUSION_G0_SGCDET_TRAIN_PREDICTIONS:-$ROOT/results/b6_g0_sgcdet_train/$RUN_TAG/$list_scope}"
LOG_ROOT="${BOXFUSION_G0_SGCDET_TRAIN_LOG_ROOT:-$ROOT/logs/b6_g0_sgcdet_train/$RUN_TAG/$list_scope}"
DIAGNOSTICS_ROOT="${BOXFUSION_G0_SGCDET_TRAIN_DIAGNOSTICS:-$ROOT/diagnostics/b6_g0_sgcdet_train/$RUN_TAG/$list_scope/online}"
BOXER_ROOT="${BOXFUSION_G0_SGCDET_TRAIN_BOXER_DIAGNOSTICS:-$ROOT/diagnostics/b6_g0_sgcdet_train/$RUN_TAG/$list_scope/boxer}"
MANIFEST="${BOXFUSION_G0_SGCDET_TRAIN_MANIFEST:-$ROOT/manifests/b6_g0_sgcdet_train/$RUN_TAG/$list_scope/collection_manifest.json}"

SCAN_ROOT="${BOXFUSION_SCANNET_SCAN_ROOT:-/extra/ZhaoX/scannet_data/scans}"
GT_ROOT="${BOXFUSION_SCANNET_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
B5_DATASET="${BOXFUSION_G0_SGCDET_B5_DATASET:-$ROOT/datasets/scannet_b6_g0_sgcdet_b5_train_v1.npz}"
SPARSE_DATASET="${BOXFUSION_G0_SGCDET_DATASET:-$ROOT/datasets/scannet_b6_g0_sgcdet_sparse_train_v1.npz}"
POTENTIAL_REPORT="${BOXFUSION_G0_SGCDET_POTENTIAL_REPORT:-$ROOT/reports/b6_g0_sgcdet_train/potential_v1.json}"
ACTIVE_CHECKPOINT="${BOXFUSION_G0_SGCDET_ACTIVE_CHECKPOINT:-$ROOT/models/scannet_b6_g0_sgcdet_sparse_refiner_v1.pt}"
IDENTITY_CHECKPOINT="${BOXFUSION_G0_SGCDET_IDENTITY_CHECKPOINT:-$ROOT/models/scannet_b6_g0_sgcdet_sparse_refiner_identity_v1.pt}"

sgcdet_sparse_assert_train_only "$SCENE_LIST" "$VAL_LIST"
for path in "$PYTHON" "$CONFIG" "$B6_CHECKPOINT" "$YOLOE_CHECKPOINT" "$MANIFEST"; do
    sgcdet_sparse_require_file "$path" "G0 sparse-refiner training dependency"
done
for path in "$PREDICTION_ROOT" "$LOG_ROOT" "$DIAGNOSTICS_ROOT" "$BOXER_ROOT" "$SCAN_ROOT" "$GT_ROOT"; do
    sgcdet_sparse_require_directory "$path" "G0 sparse-refiner training input"
done
if [[ "$ACTIVE_CHECKPOINT" == "$IDENTITY_CHECKPOINT" ]]; then
    echo "Active and identity checkpoints must use different paths." >&2
    exit 1
fi
if [[ ( -e "$ACTIVE_CHECKPOINT" || -e "$IDENTITY_CHECKPOINT" ) \
      && "${BOXFUSION_G0_SGCDET_ALLOW_OVERWRITE:-0}" != "1" ]]; then
    echo "Refusing to overwrite an existing paired checkpoint:" >&2
    echo "  active: $ACTIVE_CHECKPOINT" >&2
    echo "  identity: $IDENTITY_CHECKPOINT" >&2
    echo "Set BOXFUSION_G0_SGCDET_ALLOW_OVERWRITE=1 only for an intentional rerun." >&2
    exit 1
fi

# Re-audit immutable collection artifacts immediately before dataset build.
"$PYTHON" "$ROOT/tools/audit_g0_sgcdet_train_collection.py" \
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
    --verify-existing

mkdir -p \
    "$(dirname "$B5_DATASET")" \
    "$(dirname "$SPARSE_DATASET")" \
    "$(dirname "$POTENTIAL_REPORT")" \
    "$(dirname "$ACTIVE_CHECKPOINT")" \
    "$(dirname "$IDENTITY_CHECKPOINT")"

echo "Building strict G0+B6 K=5 AP50 source dataset (CPU only)"
CUDA_VISIBLE_DEVICES="" "$PYTHON" \
    "$ROOT/tools/build_oriented_refiner_dataset.py" \
    --diagnostics-root "$DIAGNOSTICS_ROOT" \
    --prediction-root "$PREDICTION_ROOT" \
    --scan-root "$SCAN_ROOT" \
    --gt-root "$GT_ROOT" \
    --scene-list "$SCENE_LIST" \
    --output "$B5_DATASET" \
    --forbidden-scene-list "$VAL_LIST" \
    --objective ap50 \
    --strict-k5-diagnostics \
    --strict-provenance-profile sgcdet_sparse_observer \
    --expected-top-k-views 5 \
    --min-runtime-views 2 \
    --min-runtime-points 128 \
    --runtime-minimum-extent 0.40 \
    --min-match-iou "${BOXFUSION_G0_SGCDET_MIN_MATCH_IOU:-0.15}" \
    --improvement-epsilon "${BOXFUSION_G0_SGCDET_IMPROVEMENT_EPSILON:-0.0001}" \
    --max-center-fraction "${BOXFUSION_G0_SGCDET_MAX_CENTER_FRACTION:-0.15}" \
    --max-log-dimension-residual "${BOXFUSION_G0_SGCDET_MAX_LOG_DIMENSION_RESIDUAL:-0.22314355131420976}"

echo "Building runtime-exact SGCDet sparse dataset (CPU only)"
CUDA_VISIBLE_DEVICES="" "$PYTHON" \
    "$ROOT/tools/build_sgcdet_sparse_refiner_dataset.py" \
    --b5-dataset "$B5_DATASET" \
    --diagnostics-root "$DIAGNOSTICS_ROOT" \
    --forbidden-scene-list "$VAL_LIST" \
    --output "$SPARSE_DATASET"

echo "Auditing AP50 target potential before training"
CUDA_VISIBLE_DEVICES="" "$PYTHON" \
    "$ROOT/tools/audit_sgcdet_training_potential.py" \
    --input "$SPARSE_DATASET" \
    --output "$POTENTIAL_REPORT" \
    --validation-fraction "${BOXFUSION_G0_SGCDET_VALIDATION_FRACTION:-0.20}" \
    --seed "${BOXFUSION_G0_SGCDET_SEED:-1337}"

COMMON_TRAIN_ARGS=(
    --input "$SPARSE_DATASET"
    --epochs "${BOXFUSION_G0_SGCDET_EPOCHS:-60}"
    --batch-size "${BOXFUSION_G0_SGCDET_BATCH_SIZE:-16}"
    --learning-rate "${BOXFUSION_G0_SGCDET_LR:-0.001}"
    --weight-decay "${BOXFUSION_G0_SGCDET_WEIGHT_DECAY:-0.00001}"
    --validation-fraction "${BOXFUSION_G0_SGCDET_VALIDATION_FRACTION:-0.20}"
    --seed "${BOXFUSION_G0_SGCDET_SEED:-1337}"
    --collection-manifest "$MANIFEST"
)

echo "Training AP50-primary active checkpoint (CPU only)"
CUDA_VISIBLE_DEVICES="" "$PYTHON" \
    "$ROOT/tools/train_sgcdet_sparse_refiner.py" \
    "${COMMON_TRAIN_ARGS[@]}" \
    --selection-metric ap50_proxy \
    --minimum-validation-tp50-proxy 0.000001 \
    --minimum-validation-cross-success 1 \
    --maximum-validation-drop50-rate 0.01 \
    --cross-oversample-factor "${BOXFUSION_G0_SGCDET_CROSS_OVERSAMPLE:-4}" \
    --patience "${BOXFUSION_G0_SGCDET_PATIENCE:-8}" \
    --cross-iou50-weight "${BOXFUSION_G0_SGCDET_CROSS_WEIGHT:-8.0}" \
    --preserve-iou50-weight "${BOXFUSION_G0_SGCDET_PRESERVE_WEIGHT:-4.0}" \
    --output "$ACTIVE_CHECKPOINT"

echo "Writing paired identity-control checkpoint (CPU only)"
CUDA_VISIBLE_DEVICES="" "$PYTHON" \
    "$ROOT/tools/train_sgcdet_sparse_refiner.py" \
    "${COMMON_TRAIN_ARGS[@]}" \
    --identity-only \
    --output "$IDENTITY_CHECKPOINT"

echo "G0 source dataset: $B5_DATASET"
echo "Sparse training dataset: $SPARSE_DATASET"
echo "Potential report: $POTENTIAL_REPORT"
echo "Active checkpoint: $ACTIVE_CHECKPOINT"
echo "Identity checkpoint: $IDENTITY_CHECKPOINT"
