#!/usr/bin/env bash
set -euo pipefail

# Build leakage-safe, object-local ScanNet supervision and train B5-v2.
# This script is CPU-only so it can run without occupying inference GPUs.
# Diagnostics and predictions must come from the same no-op K=5 train-scene
# memory run documented in docs/b5_local_box_refiner.md.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"
DIAGNOSTICS_ROOT="${BOXFUSION_B5V2_TRAIN_DIAGNOSTICS:-$ROOT/diagnostics/b5v2_memory_observer_train}"
PREDICTION_ROOT="${BOXFUSION_B5V2_TRAIN_PREDICTIONS:-$ROOT/results/b5v2_memory_observer_train}"
SCAN_ROOT="${BOXFUSION_SCANNET_SCAN_ROOT:-/extra/ZhaoX/scannet_data/scans}"
GT_ROOT="${BOXFUSION_SCANNET_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCENE_LIST="${BOXFUSION_B5V2_TRAIN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
DATASET="${BOXFUSION_B5V2_DATASET:-$ROOT/datasets/scannet_b5v2_oriented_refiner_train.npz}"
CHECKPOINT="${BOXFUSION_B5V2_REFINER_CHECKPOINT:-${BOXFUSION_B5V2_CHECKPOINT:-$ROOT/models/scannet_b5v2_oriented_refiner.pt}}"
MAX_CENTER_FRACTION="${BOXFUSION_B5V2_MAX_CENTER_FRACTION:-0.15}"
MAX_LOG_DIMENSION_RESIDUAL="${BOXFUSION_B5V2_MAX_LOG_DIMENSION_RESIDUAL:-0.22314355131420976}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing B5-v2 Python environment: $PYTHON" >&2
    exit 1
fi
for directory in "$DIAGNOSTICS_ROOT" "$PREDICTION_ROOT" "$SCAN_ROOT" "$GT_ROOT"; do
    if [[ ! -d "$directory" ]]; then
        echo "Missing B5-v2 training input directory: $directory" >&2
        echo "See: $ROOT/docs/b5_local_box_refiner.md" >&2
        exit 1
    fi
done
if [[ ! -f "$SCENE_LIST" ]]; then
    echo "Missing B5-v2 training scene list: $SCENE_LIST" >&2
    exit 1
fi
if [[ "$(basename "$SCENE_LIST")" == *val* ]]; then
    echo "Refusing validation-labelled B5-v2 training split: $SCENE_LIST" >&2
    echo "Use official ScanNet train scenes; never train on the headline val set." >&2
    exit 1
fi
if [[ ! -f "$ROOT/tools/build_oriented_refiner_dataset.py" \
      || ! -f "$ROOT/tools/train_oriented_box_refiner.py" ]]; then
    echo "Missing B5-v2 dataset/training tools" >&2
    exit 1
fi

mkdir -p "$(dirname "$DATASET")" "$(dirname "$CHECKPOINT")"

CUDA_VISIBLE_DEVICES="" "$PYTHON" \
    "$ROOT/tools/build_oriented_refiner_dataset.py" \
    --diagnostics-root "$DIAGNOSTICS_ROOT" \
    --prediction-root "$PREDICTION_ROOT" \
    --scan-root "$SCAN_ROOT" \
    --gt-root "$GT_ROOT" \
    --scene-list "$SCENE_LIST" \
    --output "$DATASET" \
    --min-match-iou "${BOXFUSION_B5V2_MIN_MATCH_IOU:-0.15}" \
    --improvement-epsilon "${BOXFUSION_B5V2_IMPROVEMENT_EPSILON:-0.0001}" \
    --max-center-fraction "$MAX_CENTER_FRACTION" \
    --max-log-dimension-residual "$MAX_LOG_DIMENSION_RESIDUAL"

CUDA_VISIBLE_DEVICES="" "$PYTHON" \
    "$ROOT/tools/train_oriented_box_refiner.py" \
    --input "$DATASET" \
    --output "$CHECKPOINT" \
    --epochs "${BOXFUSION_B5V2_EPOCHS:-60}" \
    --batch-size "${BOXFUSION_B5V2_BATCH_SIZE:-32}" \
    --learning-rate "${BOXFUSION_B5V2_LR:-0.001}" \
    --weight-decay "${BOXFUSION_B5V2_WEIGHT_DECAY:-0.00001}" \
    --validation-fraction "${BOXFUSION_B5V2_VALIDATION_FRACTION:-0.20}" \
    --seed "${BOXFUSION_B5V2_SEED:-1337}" \
    --point-hidden-dim "${BOXFUSION_B5V2_POINT_HIDDEN_DIM:-64}" \
    --point-embedding-dim "${BOXFUSION_B5V2_POINT_EMBEDDING_DIM:-128}" \
    --head-hidden-dim "${BOXFUSION_B5V2_HEAD_HIDDEN_DIM:-128}" \
    --max-center-fraction "$MAX_CENTER_FRACTION" \
    --max-log-dimension-residual "$MAX_LOG_DIMENSION_RESIDUAL" \
    --center-weight "${BOXFUSION_B5V2_CENTER_WEIGHT:-1.0}" \
    --dimension-weight "${BOXFUSION_B5V2_DIMENSION_WEIGHT:-1.0}" \
    --quality-weight "${BOXFUSION_B5V2_QUALITY_WEIGHT:-1.0}"

echo "B5-v2 dataset: $DATASET"
echo "B5-v2 checkpoint: $CHECKPOINT"
