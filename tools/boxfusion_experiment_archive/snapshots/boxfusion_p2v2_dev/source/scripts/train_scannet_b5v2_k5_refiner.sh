#!/usr/bin/env bash
set -euo pipefail

# Leakage-safe K5 control: train the original B5-v2 improvement objective on
# the newly collected strict K=5 diagnostics.  CPU-only.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"
COLLECT_TAG="${BOXFUSION_B5V3_K5_COLLECT_TAG:-b5v3_k5_gatealigned_train_extent040_v2}"
DIAGNOSTICS_ROOT="${BOXFUSION_B5V3_K5_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/$COLLECT_TAG}"
PREDICTION_ROOT="${BOXFUSION_B5V3_K5_PRED_ROOT:-$ROOT/results/$COLLECT_TAG}"
SCAN_ROOT="${BOXFUSION_SCANNET_SCAN_ROOT:-/extra/ZhaoX/scannet_data/scans}"
GT_ROOT="${BOXFUSION_SCANNET_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCENE_LIST="${BOXFUSION_B5V3_TRAIN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
VAL_LIST="${BOXFUSION_B5V3_FORBIDDEN_VAL_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
DATASET="${BOXFUSION_B5V2_K5_DATASET:-$ROOT/datasets/scannet_b5v2_k5_gatealigned_extent040_train_v2.npz}"
CHECKPOINT="${BOXFUSION_B5V2_K5_CHECKPOINT:-$ROOT/models/scannet_b5v2_k5_gatealigned_extent040_refiner_v2.pt}"
MIN_EXTENT="${BOXFUSION_B5V3_RUNTIME_MIN_EXTENT:-0.40}"
MAX_CENTER_FRACTION="${BOXFUSION_B5V2_K5_MAX_CENTER_FRACTION:-0.15}"
MAX_LOG_DIMENSION_RESIDUAL="${BOXFUSION_B5V2_K5_MAX_LOG_DIMENSION_RESIDUAL:-0.22314355131420976}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing B5-v2 K5 Python environment: $PYTHON" >&2
    exit 1
fi
for directory in "$DIAGNOSTICS_ROOT" "$PREDICTION_ROOT" "$SCAN_ROOT" "$GT_ROOT"; do
    if [[ ! -d "$directory" ]]; then
        echo "Missing B5-v2 K5 input directory: $directory" >&2
        exit 1
    fi
done
for file in "$SCENE_LIST" "$VAL_LIST"; do
    if [[ ! -f "$file" ]]; then
        echo "Missing B5-v2 K5 split file: $file" >&2
        exit 1
    fi
done
if [[ "${SCENE_LIST,,}" == *val* ]]; then
    echo "Refusing validation-labelled B5-v2 K5 training list: $SCENE_LIST" >&2
    exit 1
fi
overlap="$(
    awk '
        NR == FNR { if (NF) forbidden[$1] = 1; next }
        NF && ($1 in forbidden) { print $1; exit }
    ' "$VAL_LIST" "$SCENE_LIST"
)"
if [[ -n "$overlap" ]]; then
    echo "Refusing train/validation leakage; scene appears in val: $overlap" >&2
    exit 1
fi
if [[ -e "$CHECKPOINT" && "${BOXFUSION_B5V2_K5_ALLOW_OVERWRITE:-0}" != "1" ]]; then
    echo "Refusing to overwrite K5 checkpoint: $CHECKPOINT" >&2
    echo "Choose a new BOXFUSION_B5V2_K5_CHECKPOINT, or explicitly set" >&2
    echo "BOXFUSION_B5V2_K5_ALLOW_OVERWRITE=1." >&2
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
    --forbidden-scene-list "$VAL_LIST" \
    --output "$DATASET" \
    --objective improvement \
    --strict-k5-diagnostics \
    --expected-top-k-views 5 \
    --min-runtime-views 2 \
    --min-runtime-points 128 \
    --runtime-minimum-extent "$MIN_EXTENT" \
    --min-match-iou "${BOXFUSION_B5V2_K5_MIN_MATCH_IOU:-0.15}" \
    --improvement-epsilon "${BOXFUSION_B5V2_K5_IMPROVEMENT_EPSILON:-0.0001}" \
    --max-center-fraction "$MAX_CENTER_FRACTION" \
    --max-log-dimension-residual "$MAX_LOG_DIMENSION_RESIDUAL"

CUDA_VISIBLE_DEVICES="" "$PYTHON" \
    "$ROOT/tools/train_oriented_box_refiner.py" \
    --input "$DATASET" \
    --output "$CHECKPOINT" \
    --objective improvement \
    --epochs "${BOXFUSION_B5V2_K5_EPOCHS:-60}" \
    --batch-size "${BOXFUSION_B5V2_K5_BATCH_SIZE:-32}" \
    --learning-rate "${BOXFUSION_B5V2_K5_LR:-0.001}" \
    --weight-decay "${BOXFUSION_B5V2_K5_WEIGHT_DECAY:-0.00001}" \
    --validation-fraction "${BOXFUSION_B5V2_K5_VALIDATION_FRACTION:-0.20}" \
    --seed "${BOXFUSION_B5V2_K5_SEED:-1337}" \
    --point-hidden-dim "${BOXFUSION_B5V2_K5_POINT_HIDDEN_DIM:-64}" \
    --point-embedding-dim "${BOXFUSION_B5V2_K5_POINT_EMBEDDING_DIM:-128}" \
    --head-hidden-dim "${BOXFUSION_B5V2_K5_HEAD_HIDDEN_DIM:-128}" \
    --max-center-fraction "$MAX_CENTER_FRACTION" \
    --max-log-dimension-residual "$MAX_LOG_DIMENSION_RESIDUAL" \
    --center-weight "${BOXFUSION_B5V2_K5_CENTER_WEIGHT:-1.0}" \
    --dimension-weight "${BOXFUSION_B5V2_K5_DIMENSION_WEIGHT:-1.0}" \
    --quality-weight "${BOXFUSION_B5V2_K5_QUALITY_WEIGHT:-1.0}"

echo "B5-v2 strict-K5 dataset: $DATASET"
echo "B5-v2 strict-K5 checkpoint: $CHECKPOINT"
