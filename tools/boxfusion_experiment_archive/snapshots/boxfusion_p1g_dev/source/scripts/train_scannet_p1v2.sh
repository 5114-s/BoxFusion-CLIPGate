#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_ROOT="${BOXFUSION_P1V2_SOURCE_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_p1_dev}"
SOURCE_TAG="${BOXFUSION_P1V2_SOURCE_TAG:-p1_residual_inputs_train100_v1}"
DIAGNOSTICS="${BOXFUSION_P1V2_DIAGNOSTICS:-$SOURCE_ROOT/diagnostics/p1_training/$SOURCE_TAG}"
PREDICTIONS="${BOXFUSION_P1V2_PREDICTIONS:-$SOURCE_ROOT/results/p1_training/$SOURCE_TAG}"
TRAIN_LIST="${BOXFUSION_P1V2_TRAIN_SCENES:-$SOURCE_ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
FORBIDDEN_LIST="${BOXFUSION_P1V2_FORBIDDEN_SCENES:-$SOURCE_ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
GT_ROOT="${BOXFUSION_P1V2_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCANS_ROOT="${BOXFUSION_P1V2_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
B6_CHECKPOINT="${BOXFUSION_P_B6_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
SOURCE_P1_CHECKPOINT="${BOXFUSION_P1V2_SOURCE_CHECKPOINT:-$SOURCE_ROOT/models/scannet_p1_residual.pt}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-${CONDA_PREFIX:-/home/admin1/miniconda3/envs/boxfusion2}}"
PYTHON="$ENV_ROOT/bin/python"
REQUESTED="${1:-${BOXFUSION_P1V2_VARIANT:-both}}"

case "${REQUESTED^^}" in
    P1R)
        VARIANTS=(P1R)
        ;;
    P1S)
        VARIANTS=(P1S)
        ;;
    BOTH)
        VARIANTS=(P1R P1S)
        ;;
    *)
        echo "Usage: bash scripts/train_scannet_p1v2.sh [P1R|P1S|both]" >&2
        exit 2
        ;;
esac

for required_file in \
    "$PYTHON" "$TRAIN_LIST" "$FORBIDDEN_LIST" "$B6_CHECKPOINT" \
    "$SOURCE_P1_CHECKPOINT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Missing required file: $required_file" >&2
        exit 1
    fi
done
for required_directory in "$DIAGNOSTICS" "$PREDICTIONS" "$GT_ROOT" "$SCANS_ROOT"; do
    if [[ ! -d "$required_directory" ]]; then
        echo "Missing required directory: $required_directory" >&2
        exit 1
    fi
done

echo "P1-v2 controlled training"
echo "  variants: ${VARIANTS[*]}"
echo "  frozen diagnostics: $DIAGNOSTICS"
echo "  frozen predictions: $PREDICTIONS"
echo "  train scenes: $TRAIN_LIST"
echo "  forbidden scenes: $FORBIDDEN_LIST"
echo "  source provenance witness: $SOURCE_P1_CHECKPOINT"
echo "  target scope: snapshot_inside_only"
echo "  hidden dimension: ${BOXFUSION_P1V2_HIDDEN_DIM:-64}"
echo "This command trains checkpoints only; it does not change BoxFusion outputs."

for variant in "${VARIANTS[@]}"; do
    variant_lower="${variant,,}"
    if [[ "$variant" == "P1R" ]]; then
        output_default="$ROOT/models/scannet_p1r_snapshot_inside.pt"
    else
        output_default="$ROOT/models/scannet_p1s_native_sparse.pt"
    fi
    summary_default="$ROOT/reports/p1v2_training/${variant_lower}_summary.json"
    output_variable="BOXFUSION_${variant}_OUTPUT"
    summary_variable="BOXFUSION_${variant}_SUMMARY"
    output="${!output_variable:-$output_default}"
    summary="${!summary_variable:-$summary_default}"
    echo "Training $variant -> $output"
    "$PYTHON" "$ROOT/tools/train_p1v2_residual_head.py" \
        --variant "$variant" \
        --diagnostics-root "$DIAGNOSTICS" \
        --prediction-root "$PREDICTIONS" \
        --gt-root "$GT_ROOT" \
        --scans-root "$SCANS_ROOT" \
        --train-scene-list "$TRAIN_LIST" \
        --forbidden-scene-list "$FORBIDDEN_LIST" \
        --b6-checkpoint "$B6_CHECKPOINT" \
        --source-p1-checkpoint "$SOURCE_P1_CHECKPOINT" \
        --output "$output" \
        --covered-iou "${BOXFUSION_P1V2_COVERED_IOU:-0.15}" \
        --assignment-topk "${BOXFUSION_P1V2_ASSIGNMENT_TOPK:-6}" \
        --negative-ratio "${BOXFUSION_P1V2_NEGATIVE_RATIO:-8.0}" \
        --maximum-loss-voxels-per-snapshot "${BOXFUSION_P1V2_MAXIMUM_LOSS_VOXELS_PER_SNAPSHOT:-4096}" \
        --hidden-dim "${BOXFUSION_P1V2_HIDDEN_DIM:-64}" \
        --validation-fraction "${BOXFUSION_P1V2_VALIDATION_FRACTION:-0.20}" \
        --epochs "${BOXFUSION_P1V2_EPOCHS:-80}" \
        --learning-rate "${BOXFUSION_P1V2_LEARNING_RATE:-0.001}" \
        --weight-decay "${BOXFUSION_P1V2_WEIGHT_DECAY:-0.0001}" \
        --regression-weight "${BOXFUSION_P1V2_REGRESSION_WEIGHT:-1.0}" \
        --snapshots-per-optimizer-step "${BOXFUSION_P1V2_SNAPSHOTS_PER_OPTIMIZER_STEP:-4}" \
        --seed "${BOXFUSION_P1V2_SEED:-1337}" \
        --device "${BOXFUSION_P1V2_DEVICE:-cpu}" \
        --summary-json "$summary"
done
