#!/usr/bin/env bash
set -euo pipefail

# Build strict train-only supervision and train the joint B3 -> B5 + B6-v2
# local head.  All commands are CPU-only.  "validation-fraction" below means a
# deterministic scene-held-out subset of ScanNet TRAIN, never ScanNet val.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"

COLLECT_TAG="${BOXFUSION_JOINT_B356_COLLECT_TAG:-joint_b356_k5_p128_observer_train_v1}"
DIAGNOSTICS_ROOT="${BOXFUSION_JOINT_B356_TRAIN_DIAGNOSTICS:-$ROOT/diagnostics/$COLLECT_TAG}"
PREDICTION_ROOT="${BOXFUSION_JOINT_B356_TRAIN_PRED_ROOT:-$ROOT/results/$COLLECT_TAG}"
SCAN_ROOT="${BOXFUSION_SCANNET_SCAN_ROOT:-/extra/ZhaoX/scannet_data/scans}"
GT_ROOT="${BOXFUSION_SCANNET_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCENE_LIST="${BOXFUSION_JOINT_B356_TRAIN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
VAL_LIST="${BOXFUSION_JOINT_B356_FORBIDDEN_VAL_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"

B5_DATASET="${BOXFUSION_JOINT_B356_B5_DATASET:-$ROOT/datasets/scannet_joint_b356_b5_ap50_train_v1.npz}"
JOINT_DATASET="${BOXFUSION_JOINT_B356_DATASET:-$ROOT/datasets/scannet_joint_b356_k5_p128_train_v1.npz}"
CHECKPOINT="${BOXFUSION_JOINT_B356_CHECKPOINT:-$ROOT/models/scannet_joint_b356_k5_p128_v1.pt}"
MIN_EXTENT="${BOXFUSION_JOINT_B356_MIN_EXTENT:-0.40}"
MAX_CENTER_FRACTION="${BOXFUSION_JOINT_B356_MAX_CENTER_FRACTION:-0.15}"
MAX_LOG_DIMENSION_RESIDUAL="${BOXFUSION_JOINT_B356_MAX_LOG_DIMENSION_RESIDUAL:-0.22314355131420976}"
RESUME_TRAIN="${BOXFUSION_JOINT_B356_RESUME_TRAIN:-0}"

if [[ "$RESUME_TRAIN" != "0" && "$RESUME_TRAIN" != "1" ]]; then
    echo "BOXFUSION_JOINT_B356_RESUME_TRAIN must be 0 or 1" >&2
    exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing joint training Python environment: $PYTHON" >&2
    exit 1
fi
for directory in "$DIAGNOSTICS_ROOT" "$PREDICTION_ROOT" "$SCAN_ROOT" "$GT_ROOT"; do
    if [[ ! -d "$directory" ]]; then
        echo "Missing joint training input directory: $directory" >&2
        exit 1
    fi
done
for file in "$SCENE_LIST" "$VAL_LIST"; do
    if [[ ! -s "$file" ]]; then
        echo "Missing or empty joint split file: $file" >&2
        exit 1
    fi
done
scene_list_name="$(basename -- "$SCENE_LIST")"
scene_list_name_lower="${scene_list_name,,}"
if [[ "$scene_list_name_lower" =~ (^|[_-])(val|validation)([_.-]|$) ]]; then
    echo "Refusing validation-labelled joint training list: $SCENE_LIST" >&2
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
duplicate="$(
    awk 'NF { count[$1] += 1; if (count[$1] == 2) { print $1; exit } }' \
        "$SCENE_LIST"
)"
if [[ -n "$duplicate" ]]; then
    echo "Refusing duplicate train scene in list: $duplicate" >&2
    exit 1
fi

while IFS= read -r scene || [[ -n "$scene" ]]; do
    [[ -z "$scene" ]] && continue
    prediction="$PREDICTION_ROOT/${scene}_boxes.pkl"
    diagnostic="$DIAGNOSTICS_ROOT/${scene}_tracks.npz"
    if [[ ! -s "$prediction" || ! -s "$diagnostic" ]]; then
        echo "Missing complete train-only prediction/diagnostic pair: $scene" >&2
        echo "Expected: $prediction" >&2
        echo "Expected: $diagnostic" >&2
        echo "Run scripts/collect_scannet_joint_b356_train.sh first." >&2
        exit 1
    fi
done <"$SCENE_LIST"

if [[ -e "$CHECKPOINT" ]]; then
    echo "Refusing to overwrite joint checkpoint: $CHECKPOINT" >&2
    exit 1
fi
if [[ -e "$JOINT_DATASET" && ! -e "$B5_DATASET" ]]; then
    echo "Refusing orphan joint dataset without its B5 source: $JOINT_DATASET" >&2
    exit 1
fi
if [[ "$RESUME_TRAIN" != "1" ]]; then
    for output in "$B5_DATASET" "$JOINT_DATASET"; do
        if [[ -e "$output" ]]; then
            echo "Refusing to reuse joint training artifact: $output" >&2
            echo "Set BOXFUSION_JOINT_B356_RESUME_TRAIN=1 for a strict" >&2
            echo "validated resume, or choose fresh artifact paths." >&2
            exit 1
        fi
    done
fi

mkdir -p \
    "$(dirname "$B5_DATASET")" \
    "$(dirname "$JOINT_DATASET")" \
    "$(dirname "$CHECKPOINT")"

if [[ ! -e "$B5_DATASET" ]]; then
    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
        "$ROOT/tools/build_oriented_refiner_dataset.py" \
        --diagnostics-root "$DIAGNOSTICS_ROOT" \
        --prediction-root "$PREDICTION_ROOT" \
        --scan-root "$SCAN_ROOT" \
        --gt-root "$GT_ROOT" \
        --scene-list "$SCENE_LIST" \
        --forbidden-scene-list "$VAL_LIST" \
        --output "$B5_DATASET" \
        --objective ap50 \
        --strict-k5-diagnostics \
        --expected-top-k-views 5 \
        --min-runtime-views 2 \
        --min-runtime-points 128 \
        --runtime-minimum-extent "$MIN_EXTENT" \
        --near-iou50-band "${BOXFUSION_JOINT_B356_NEAR_IOU50_BAND:-0.15}" \
        --gain-cap "${BOXFUSION_JOINT_B356_GAIN_CAP:-0.25}" \
        --gain-sample-weight "${BOXFUSION_JOINT_B356_GAIN_SAMPLE_WEIGHT:-2.0}" \
        --cross-iou50-sample-weight "${BOXFUSION_JOINT_B356_CROSS_SAMPLE_WEIGHT:-4.0}" \
        --near-iou50-sample-weight "${BOXFUSION_JOINT_B356_NEAR_SAMPLE_WEIGHT:-2.0}" \
        --min-match-iou "${BOXFUSION_JOINT_B356_MIN_MATCH_IOU:-0.15}" \
        --improvement-epsilon "${BOXFUSION_JOINT_B356_IMPROVEMENT_EPSILON:-0.0001}" \
        --max-center-fraction "$MAX_CENTER_FRACTION" \
        --max-log-dimension-residual "$MAX_LOG_DIMENSION_RESIDUAL"
else
    echo "Strict resume: reusing existing B5 source $B5_DATASET"
fi

if [[ ! -e "$JOINT_DATASET" ]]; then
    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
        "$ROOT/tools/build_joint_local_dataset.py" \
        --b5-dataset "$B5_DATASET" \
        --diagnostics-root "$DIAGNOSTICS_ROOT" \
        --forbidden-scene-list "$VAL_LIST" \
        --output "$JOINT_DATASET"
else
    stored_source_sha="$(
        PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -c \
            'import sys; sys.path.insert(0, sys.argv[1]); from tools.train_joint_local_head import load_joint_local_dataset; print(load_joint_local_dataset(sys.argv[2]).source_dataset_sha256)' \
            "$ROOT" "$JOINT_DATASET"
    )"
    actual_source_sha="$(sha256sum "$B5_DATASET" | awk '{print $1}')"
    if [[ "$stored_source_sha" != "$actual_source_sha" ]]; then
        echo "Refusing resume: joint dataset B5 SHA-256 mismatch" >&2
        exit 1
    fi
    echo "Strict resume: validated existing joint dataset $JOINT_DATASET"
fi

CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
    "$ROOT/tools/train_joint_local_head.py" \
    --input "$JOINT_DATASET" \
    --output "$CHECKPOINT" \
    --epochs "${BOXFUSION_JOINT_B356_EPOCHS:-100}" \
    --batch-size "${BOXFUSION_JOINT_B356_BATCH_SIZE:-32}" \
    --learning-rate "${BOXFUSION_JOINT_B356_LR:-0.001}" \
    --weight-decay "${BOXFUSION_JOINT_B356_WEIGHT_DECAY:-0.00001}" \
    --validation-fraction "${BOXFUSION_JOINT_B356_TRAIN_HOLDOUT_FRACTION:-0.20}" \
    --seed "${BOXFUSION_JOINT_B356_SEED:-1337}" \
    --point-hidden-dim "${BOXFUSION_JOINT_B356_POINT_HIDDEN_DIM:-48}" \
    --point-embedding-dim "${BOXFUSION_JOINT_B356_POINT_EMBEDDING_DIM:-96}" \
    --view-embedding-dim "${BOXFUSION_JOINT_B356_VIEW_EMBEDDING_DIM:-96}" \
    --head-hidden-dim "${BOXFUSION_JOINT_B356_HEAD_HIDDEN_DIM:-128}" \
    --max-center-fraction "$MAX_CENTER_FRACTION" \
    --max-log-dimension-residual "$MAX_LOG_DIMENSION_RESIDUAL" \
    --center-weight "${BOXFUSION_JOINT_B356_CENTER_WEIGHT:-1.0}" \
    --dimension-weight "${BOXFUSION_JOINT_B356_DIMENSION_WEIGHT:-1.0}" \
    --identity-weight "${BOXFUSION_JOINT_B356_IDENTITY_WEIGHT:-0.25}" \
    --improvement-weight "${BOXFUSION_JOINT_B356_IMPROVEMENT_WEIGHT:-1.0}" \
    --iou-gain-weight "${BOXFUSION_JOINT_B356_IOU_GAIN_WEIGHT:-2.0}" \
    --cross-iou50-weight "${BOXFUSION_JOINT_B356_CROSS_IOU50_WEIGHT:-4.0}" \
    --preserve-iou50-weight "${BOXFUSION_JOINT_B356_PRESERVE_IOU50_WEIGHT:-2.0}" \
    --dual-iou-weight "${BOXFUSION_JOINT_B356_DUAL_IOU_WEIGHT:-1.0}" \
    --ordinal-weight "${BOXFUSION_JOINT_B356_ORDINAL_WEIGHT:-1.0}" \
    --uncertainty-weight "${BOXFUSION_JOINT_B356_UNCERTAINTY_WEIGHT:-0.10}"

echo "Joint B5 supervision dataset: $B5_DATASET"
echo "Joint multi-view dataset: $JOINT_DATASET"
echo "Joint B3/B5/B6-v2 checkpoint: $CHECKPOINT"
