#!/usr/bin/env bash
set -euo pipefail

# CPU-only, leakage-safe construction and training for the local sparse head.
# Collection is a separate explicit GPU step; this script never starts CUDA.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/lib_sgcdet_sparse_protocol.sh
source "$SCRIPT_DIR/lib_sgcdet_sparse_protocol.sh"

ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
RUN_TAG="${BOXFUSION_SGCDET_COLLECT_TAG:-sgcdet_sparse_observer_train_v1}"
DIAGNOSTICS_ROOT="${BOXFUSION_SGCDET_TRAIN_DIAGNOSTICS:-$ROOT/diagnostics/$RUN_TAG}"
PREDICTION_ROOT="${BOXFUSION_SGCDET_TRAIN_PREDICTIONS:-$ROOT/results/$RUN_TAG}"
SCENE_LIST="${BOXFUSION_SGCDET_TRAIN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
VAL_LIST="${BOXFUSION_SGCDET_VAL_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
SCAN_ROOT="${BOXFUSION_SCANNET_SCAN_ROOT:-/extra/ZhaoX/scannet_data/scans}"
GT_ROOT="${BOXFUSION_SCANNET_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"

B5_DATASET="${BOXFUSION_SGCDET_B5_DATASET:-$ROOT/datasets/scannet_sgcdet_sparse_b5_source_train.npz}"
SPARSE_DATASET="${BOXFUSION_SGCDET_DATASET:-$ROOT/datasets/scannet_sgcdet_sparse_refiner_train.npz}"
ACTIVE_CHECKPOINT="${BOXFUSION_SGCDET_SPARSE_CHECKPOINT:-$ROOT/models/scannet_sgcdet_sparse_refiner.pt}"
IDENTITY_CHECKPOINT="${BOXFUSION_SGCDET_IDENTITY_CHECKPOINT:-$ROOT/models/scannet_sgcdet_sparse_refiner_identity.pt}"

sgcdet_sparse_require_file "$PYTHON" "BoxFusion training Python"
sgcdet_sparse_assert_train_only "$SCENE_LIST" "$VAL_LIST"
sgcdet_sparse_require_directory "$DIAGNOSTICS_ROOT" "sparse observer train diagnostics"
sgcdet_sparse_require_directory "$PREDICTION_ROOT" "sparse observer train predictions"
sgcdet_sparse_require_directory "$SCAN_ROOT" "ScanNet train scans"
sgcdet_sparse_require_directory "$GT_ROOT" "ScanNet train detection ground truth"

for tool in \
    "$ROOT/tools/build_oriented_refiner_dataset.py" \
    "$ROOT/tools/build_sgcdet_sparse_refiner_dataset.py" \
    "$ROOT/tools/train_sgcdet_sparse_refiner.py"; do
    sgcdet_sparse_require_file "$tool" "sparse-refiner training tool"
done

if [[ "$ACTIVE_CHECKPOINT" == "$IDENTITY_CHECKPOINT" ]]; then
    echo "Active and identity sparse checkpoints must use different paths." >&2
    exit 1
fi

mkdir -p \
    "$(dirname "$B5_DATASET")" \
    "$(dirname "$SPARSE_DATASET")" \
    "$(dirname "$ACTIVE_CHECKPOINT")" \
    "$(dirname "$IDENTITY_CHECKPOINT")"

echo "Building strict K=5 source dataset (CPU only)"
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
    --expected-top-k-views 5 \
    --min-runtime-views 2 \
    --min-runtime-points 128 \
    --runtime-minimum-extent 0.40 \
    --min-match-iou "${BOXFUSION_SGCDET_MIN_MATCH_IOU:-0.15}" \
    --improvement-epsilon "${BOXFUSION_SGCDET_IMPROVEMENT_EPSILON:-0.0001}" \
    --max-center-fraction "${BOXFUSION_SGCDET_MAX_CENTER_FRACTION:-0.15}" \
    --max-log-dimension-residual "${BOXFUSION_SGCDET_MAX_LOG_DIMENSION_RESIDUAL:-0.22314355131420976}"

echo "Building strict SGCDet-inspired sparse dataset (CPU only)"
CUDA_VISIBLE_DEVICES="" "$PYTHON" \
    "$ROOT/tools/build_sgcdet_sparse_refiner_dataset.py" \
    --b5-dataset "$B5_DATASET" \
    --diagnostics-root "$DIAGNOSTICS_ROOT" \
    --forbidden-scene-list "$VAL_LIST" \
    --output "$SPARSE_DATASET"

COMMON_TRAIN_ARGS=(
    --input "$SPARSE_DATASET"
    --epochs "${BOXFUSION_SGCDET_EPOCHS:-40}"
    --batch-size "${BOXFUSION_SGCDET_BATCH_SIZE:-16}"
    --learning-rate "${BOXFUSION_SGCDET_LR:-0.001}"
    --weight-decay "${BOXFUSION_SGCDET_WEIGHT_DECAY:-0.00001}"
    --validation-fraction "${BOXFUSION_SGCDET_VALIDATION_FRACTION:-0.20}"
    --seed "${BOXFUSION_SGCDET_SEED:-1337}"
)

echo "Training active sparse-refiner checkpoint (CPU only)"
CUDA_VISIBLE_DEVICES="" "$PYTHON" \
    "$ROOT/tools/train_sgcdet_sparse_refiner.py" \
    "${COMMON_TRAIN_ARGS[@]}" \
    --output "$ACTIVE_CHECKPOINT"

echo "Writing strict identity-control checkpoint (CPU only)"
CUDA_VISIBLE_DEVICES="" "$PYTHON" \
    "$ROOT/tools/train_sgcdet_sparse_refiner.py" \
    "${COMMON_TRAIN_ARGS[@]}" \
    --identity-only \
    --output "$IDENTITY_CHECKPOINT"

echo "Sparse source dataset: $B5_DATASET"
echo "Sparse training dataset: $SPARSE_DATASET"
echo "Active checkpoint: $ACTIVE_CHECKPOINT"
echo "Identity checkpoint: $IDENTITY_CHECKPOINT"
