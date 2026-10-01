#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_ROOT="${BOXFUSION_P1G_SOURCE_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_p1_dev}"
SOURCE_TAG="${BOXFUSION_P1G_SOURCE_TAG:-p1_residual_inputs_train100_v1}"
DIAGNOSTICS="${BOXFUSION_P1G_DIAGNOSTICS:-$SOURCE_ROOT/diagnostics/p1_training/$SOURCE_TAG}"
PREDICTIONS="${BOXFUSION_P1G_PREDICTIONS:-$SOURCE_ROOT/results/p1_training/$SOURCE_TAG}"
TRAIN_LIST="${BOXFUSION_P1G_TRAIN_SCENES:-$SOURCE_ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
FIT_LIST="${BOXFUSION_P1G_FIT_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_p1g_fit60.txt}"
CAL_LIST="${BOXFUSION_P1G_CAL_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_p1g_cal20.txt}"
AUDIT_LIST="${BOXFUSION_P1G_AUDIT_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_p1g_audit20.txt}"
FORBIDDEN_LIST="${BOXFUSION_P1G_FORBIDDEN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
GT_ROOT="${BOXFUSION_P1G_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCANS_ROOT="${BOXFUSION_P1G_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
B6_CHECKPOINT="${BOXFUSION_P1G_B6_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
SOURCE_P1_CHECKPOINT="${BOXFUSION_P1G_SOURCE_P1_CHECKPOINT:-$SOURCE_ROOT/models/scannet_p1_residual.pt}"
P1S_CHECKPOINT="${BOXFUSION_P1G_P1S_CHECKPOINT:-$ROOT/models/scannet_p1s_native_sparse.pt}"
OUTPUT="${BOXFUSION_P1G_OUTPUT:-$ROOT/models/scannet_p1g_aligned_geometry.pt}"
SUMMARY="${BOXFUSION_P1G_SUMMARY:-$ROOT/reports/p1g_training/p1g_aligned_geometry_summary.json}"
PYTHON="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}/bin/python"

for path in \
    "$PYTHON" "$TRAIN_LIST" "$FIT_LIST" "$CAL_LIST" "$AUDIT_LIST" \
    "$FORBIDDEN_LIST" "$B6_CHECKPOINT" "$SOURCE_P1_CHECKPOINT" \
    "$P1S_CHECKPOINT"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing P1G training input: $path" >&2
        exit 1
    fi
done
for path in "$DIAGNOSTICS" "$PREDICTIONS" "$GT_ROOT" "$SCANS_ROOT"; do
    if [[ ! -d "$path" ]]; then
        echo "Missing P1G training directory: $path" >&2
        exit 1
    fi
done

echo "P1G aligned-frame geometry-only training"
echo "  fit/cal/audit: 60/20/20 train-only scenes"
echo "  frozen P1S: $P1S_CHECKPOINT"
echo "  frozen diagnostics: $DIAGNOSTICS"
echo "  frozen predictions: $PREDICTIONS"
echo "  output: $OUTPUT"
echo "This command trains only the six-dimensional geometry head."

"$PYTHON" "$ROOT/tools/train_p1g_geometry_refiner.py" \
    --diagnostics-root "$DIAGNOSTICS" \
    --prediction-root "$PREDICTIONS" \
    --gt-root "$GT_ROOT" \
    --scans-root "$SCANS_ROOT" \
    --train-scene-list "$TRAIN_LIST" \
    --fit-scene-list "$FIT_LIST" \
    --cal-scene-list "$CAL_LIST" \
    --audit-scene-list "$AUDIT_LIST" \
    --full-val-scene-list "$FORBIDDEN_LIST" \
    --b6-checkpoint "$B6_CHECKPOINT" \
    --source-p1-checkpoint "$SOURCE_P1_CHECKPOINT" \
    --p1s-checkpoint "$P1S_CHECKPOINT" \
    --output "$OUTPUT" \
    --summary-json "$SUMMARY" \
    --covered-iou "${BOXFUSION_P1G_COVERED_IOU:-0.15}" \
    --assignment-topk "${BOXFUSION_P1G_ASSIGNMENT_TOPK:-6}" \
    --negative-ratio "${BOXFUSION_P1G_NEGATIVE_RATIO:-8.0}" \
    --maximum-loss-voxels-per-snapshot "${BOXFUSION_P1G_MAXIMUM_LOSS_VOXELS_PER_SNAPSHOT:-4096}" \
    --epochs "${BOXFUSION_P1G_EPOCHS:-80}" \
    --batch-size "${BOXFUSION_P1G_BATCH_SIZE:-512}" \
    --learning-rate "${BOXFUSION_P1G_LEARNING_RATE:-0.001}" \
    --weight-decay "${BOXFUSION_P1G_WEIGHT_DECAY:-0.0001}" \
    --max-center-offset "${BOXFUSION_P1G_MAX_CENTER_OFFSET:-1.0}" \
    --min-box-extent "${BOXFUSION_P1G_MIN_BOX_EXTENT:-0.08}" \
    --max-box-extent "${BOXFUSION_P1G_MAX_BOX_EXTENT:-4.0}" \
    --adapter-epsilon "${BOXFUSION_P1G_ADAPTER_EPSILON:-0.000001}" \
    --smooth-l1-weight "${BOXFUSION_P1G_SMOOTH_L1_WEIGHT:-0.1}" \
    --smooth-l1-beta "${BOXFUSION_P1G_SMOOTH_L1_BETA:-0.1}" \
    --seed "${BOXFUSION_P1G_SEED:-1337}" \
    --device "${BOXFUSION_P1G_DEVICE:-cpu}"
