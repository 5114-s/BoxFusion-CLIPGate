#!/usr/bin/env bash
set -euo pipefail

# Train P2 only from the immutable train-only P1 collection.  This script does
# not run BoxFusion inference and never reads ScanNet validation ground truth.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
P1_TAG="${BOXFUSION_P1_TRAIN_RUN_TAG:-p1_residual_inputs_train100_v1}"
P1_ARTIFACT_ROOT="${BOXFUSION_P1_ARTIFACT_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_p1_dev}"
DIAGNOSTICS="${BOXFUSION_P2_TRAIN_DIAGNOSTICS:-$P1_ARTIFACT_ROOT/diagnostics/p1_training/$P1_TAG}"
PREDICTIONS="${BOXFUSION_P2_TRAIN_PREDICTIONS:-$P1_ARTIFACT_ROOT/results/p1_training/$P1_TAG}"
TRAIN_LIST="${BOXFUSION_P2_TRAIN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
FORBIDDEN_LIST="${BOXFUSION_P2_FORBIDDEN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
GT_ROOT="${BOXFUSION_P2_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCANS_ROOT="${BOXFUSION_P2_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
P1_CHECKPOINT="${BOXFUSION_P1_RESIDUAL_CHECKPOINT:-$ROOT/models/scannet_p1_residual.pt}"
B6_CHECKPOINT="${BOXFUSION_P_B6_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
OUTPUT="${BOXFUSION_P2_OUTPUT:-$ROOT/models/scannet_p2_occupancy_topk.pt}"
SUMMARY="${BOXFUSION_P2_TRAIN_SUMMARY:-$ROOT/reports/p2_training_summary.json}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-${CONDA_PREFIX:-/home/admin1/miniconda3/envs/boxfusion2}}"
PYTHON="$ENV_ROOT/bin/python"

for path in \
    "$PYTHON" \
    "$TRAIN_LIST" \
    "$FORBIDDEN_LIST" \
    "$P1_CHECKPOINT" \
    "$B6_CHECKPOINT"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing P2 training input: $path" >&2
        exit 1
    fi
done
for directory in "$DIAGNOSTICS" "$PREDICTIONS" "$GT_ROOT" "$SCANS_ROOT"; do
    if [[ ! -d "$directory" ]]; then
        echo "Missing P2 training directory: $directory" >&2
        exit 1
    fi
done

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$SUMMARY")"
exec "$PYTHON" "$ROOT/tools/train_p2_occupancy_topk.py" \
    --diagnostics-root "$DIAGNOSTICS" \
    --prediction-root "$PREDICTIONS" \
    --gt-root "$GT_ROOT" \
    --scans-root "$SCANS_ROOT" \
    --train-scene-list "$TRAIN_LIST" \
    --forbidden-scene-list "$FORBIDDEN_LIST" \
    --p1-checkpoint "$P1_CHECKPOINT" \
    --b6-checkpoint "$B6_CHECKPOINT" \
    --output "$OUTPUT" \
    --covered-iou "${BOXFUSION_P2_COVERED_IOU:-0.15}" \
    --occupancy-margin "${BOXFUSION_P2_OCCUPANCY_MARGIN:-0.0}" \
    --max-voxels-per-scene "${BOXFUSION_P2_MAX_VOXELS_PER_SCENE:-60000}" \
    --negative-ratio "${BOXFUSION_P2_NEGATIVE_RATIO:-8.0}" \
    --hidden-dim "${BOXFUSION_P2_HIDDEN_DIM:-32}" \
    --validation-fraction "${BOXFUSION_P2_VALIDATION_FRACTION:-0.20}" \
    --epochs "${BOXFUSION_P2_EPOCHS:-100}" \
    --learning-rate "${BOXFUSION_P2_LEARNING_RATE:-0.001}" \
    --weight-decay "${BOXFUSION_P2_WEIGHT_DECAY:-0.0}" \
    --batch-size "${BOXFUSION_P2_BATCH_SIZE:-8192}" \
    --seed "${BOXFUSION_P2_TRAIN_SEED:-1337}" \
    --device "${BOXFUSION_P2_TRAIN_DEVICE:-cpu}" \
    --summary-json "$SUMMARY"
