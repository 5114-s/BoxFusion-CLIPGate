#!/usr/bin/env bash
set -euo pipefail

# Build a scene-labelled quality dataset and train the B6 multi-task scorer.
# This script is CPU-only by design and refuses a scene list whose filename
# looks like a validation split unless explicitly overridden.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"
DIAGNOSTICS_ROOT="${BOXFUSION_B6_TRAIN_DIAGNOSTICS:-$ROOT/diagnostics/b6_quality_observer_train}"
SCAN_ROOT="${BOXFUSION_SCANNET_SCAN_ROOT:-/extra/ZhaoX/scannet_data/scans}"
GT_ROOT="${BOXFUSION_SCANNET_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCENE_LIST="${BOXFUSION_B6_TRAIN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
DATASET="${BOXFUSION_B6_DATASET:-$ROOT/datasets/scannet_b6_quality_train.npz}"
CHECKPOINT="${BOXFUSION_B6_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"

if [[ -z "$DIAGNOSTICS_ROOT" || ! -d "$DIAGNOSTICS_ROOT" ]]; then
    echo "Missing B6 train-scene diagnostics: $DIAGNOSTICS_ROOT" >&2
    echo "Run: bash scripts/collect_scannet_b6_train_diagnostics.sh 0,1" >&2
    exit 1
fi
if [[ ! -f "$SCENE_LIST" ]]; then
    echo "Missing B6 training scene list: $SCENE_LIST" >&2
    exit 1
fi
if [[ "$(basename "$SCENE_LIST")" == *val* \
      && "${BOXFUSION_ALLOW_VALIDATION_TRAINING:-0}" != "1" ]]; then
    echo "Refusing validation-labelled training split: $SCENE_LIST" >&2
    echo "Use official ScanNet train scenes; do not train on the headline val set." >&2
    exit 1
fi

mkdir -p "$(dirname "$DATASET")" "$(dirname "$CHECKPOINT")"

CUDA_VISIBLE_DEVICES="" "$PYTHON" "$ROOT/tools/build_refiner_dataset.py" \
    --diagnostics-root "$DIAGNOSTICS_ROOT" \
    --scan-root "$SCAN_ROOT" \
    --gt-root "$GT_ROOT" \
    --scene-list "$SCENE_LIST" \
    --min-iou 0.15 \
    --include-negatives \
    --output "$DATASET"

CUDA_VISIBLE_DEVICES="" "$PYTHON" "$ROOT/tools/train_quality_calibrator.py" \
    --input "$DATASET" \
    --output "$CHECKPOINT" \
    --model iou_mlp \
    --target-kind iou \
    --epochs "${BOXFUSION_B6_EPOCHS:-400}" \
    --learning-rate "${BOXFUSION_B6_LR:-0.001}" \
    --l2-weight "${BOXFUSION_B6_WEIGHT_DECAY:-0.0001}" \
    --validation-fraction "${BOXFUSION_B6_VALIDATION_FRACTION:-0.20}" \
    --seed "${BOXFUSION_B6_SEED:-1337}" \
    --hidden-dims "${BOXFUSION_B6_HIDDEN_DIMS:-64,32}" \
    --ranking-weights "${BOXFUSION_B6_RANKING_WEIGHTS:-0.10,0.20,0.30,0.40}"

echo "B6 checkpoint: $CHECKPOINT"
