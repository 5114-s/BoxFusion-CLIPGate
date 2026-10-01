#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TAG="${BOXFUSION_P1_TRAIN_RUN_TAG:-p1_residual_inputs_train100_v1}"
DIAGNOSTICS="${BOXFUSION_P1_TRAIN_DIAGNOSTICS:-$ROOT/diagnostics/p1_training/$TAG}"
PREDICTIONS="${BOXFUSION_P1_TRAIN_PREDICTIONS:-$ROOT/results/p1_training/$TAG}"
TRAIN_LIST="${BOXFUSION_P1_TRAIN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
VAL_LIST="${BOXFUSION_P1_FORBIDDEN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
GT_ROOT="${BOXFUSION_P1_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCANS_ROOT="${BOXFUSION_P1_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
OUTPUT="${BOXFUSION_P1_OUTPUT:-$ROOT/models/scannet_p1_residual.pt}"
B6_CHECKPOINT="${BOXFUSION_P_B6_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
SUMMARY="${BOXFUSION_P1_TRAIN_SUMMARY:-$ROOT/reports/p1_training_summary.json}"
PYTHON="${BOXFUSION_ENV_ROOT:-${CONDA_PREFIX:-/home/admin1/miniconda3/envs/boxfusion2}}/bin/python"

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$SUMMARY")"
exec "$PYTHON" "$ROOT/tools/train_p1_residual_head.py" \
    --diagnostics-root "$DIAGNOSTICS" \
    --prediction-root "$PREDICTIONS" \
    --gt-root "$GT_ROOT" \
    --scans-root "$SCANS_ROOT" \
    --train-scene-list "$TRAIN_LIST" \
    --forbidden-scene-list "$VAL_LIST" \
    --b6-checkpoint "$B6_CHECKPOINT" \
    --output "$OUTPUT" \
    --covered-iou "${BOXFUSION_P1_COVERED_IOU:-0.15}" \
    --assignment-topk "${BOXFUSION_P1_ASSIGNMENT_TOPK:-6}" \
    --max-voxels-per-scene "${BOXFUSION_P1_MAX_VOXELS_PER_SCENE:-60000}" \
    --negative-ratio "${BOXFUSION_P1_NEGATIVE_RATIO:-8.0}" \
    --hidden-dim "${BOXFUSION_P1_HIDDEN_DIM:-64}" \
    --validation-fraction "${BOXFUSION_P1_VALIDATION_FRACTION:-0.20}" \
    --epochs "${BOXFUSION_P1_EPOCHS:-120}" \
    --learning-rate "${BOXFUSION_P1_LEARNING_RATE:-0.001}" \
    --weight-decay "${BOXFUSION_P1_WEIGHT_DECAY:-0.0001}" \
    --regression-weight "${BOXFUSION_P1_REGRESSION_WEIGHT:-1.0}" \
    --batch-size "${BOXFUSION_P1_BATCH_SIZE:-8192}" \
    --seed "${BOXFUSION_P1_TRAIN_SEED:-1337}" \
    --device "${BOXFUSION_P1_TRAIN_DEVICE:-cpu}" \
    --summary-json "$SUMMARY"
