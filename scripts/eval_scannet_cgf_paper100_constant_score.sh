#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
    echo "Usage: $0 EXPERIMENT_NAME [PREDICTION_ROOT]" >&2
    exit 2
fi

EXPERIMENT_NAME="$1"
if [[ ! "$EXPERIMENT_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid experiment name: $EXPERIMENT_NAME" >&2
    exit 2
fi

ROOT=/data/ZhaoX/BoxFusion
EVAL_CODE="$ROOT/upstream_clean/BoxFusion_shallow/evaluation"
PYTHON=/home/admin1/miniconda3/envs/boxfusion2/bin/python
PRED_ROOT="${2:-$ROOT/results/$EXPERIMENT_NAME}"
META="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
LOG_ROOT="$ROOT/logs/cgf_paper100_constant_score"
MPL_ROOT="$LOG_ROOT/mplconfig"
LOG_PATH="$LOG_ROOT/${EXPERIMENT_NAME}_constant.log"
EXPECTED_EVALUATOR_SHA=aea2a72940b7cc53ee273f9f235e2efc848e1994e22da5f439af9751e1e27c27

if [[ ! -d "$PRED_ROOT" ]]; then
    echo "Prediction root does not exist: $PRED_ROOT" >&2
    exit 1
fi
# The evaluator runs after changing into its own source directory.  Resolve the
# prediction root here so a caller-supplied relative path cannot silently turn
# into an empty evaluation (which otherwise yields NaN metrics).
PRED_ROOT=$(realpath "$PRED_ROOT")

evaluator_sha=$(sha256sum "$EVAL_CODE/eval_scannet.py" | awk '{print $1}')
if [[ "$evaluator_sha" != "$EXPECTED_EVALUATOR_SHA" ]]; then
    echo "Constant-score evaluator hash mismatch: $evaluator_sha" >&2
    exit 1
fi

expected=$(awk 'END {print NR}' "$META")
found=$(find "$PRED_ROOT" -maxdepth 1 -type f -name 'scene*_boxes.pkl' | wc -l)
if [[ "$found" -ne "$expected" ]]; then
    echo "Expected $expected predictions, found $found in $PRED_ROOT" >&2
    exit 1
fi

while IFS= read -r scene || [[ -n "$scene" ]]; do
    if [[ ! -s "$PRED_ROOT/${scene}_boxes.pkl" ]]; then
        echo "Missing prediction for official scene: $scene" >&2
        exit 1
    fi
done < "$META"

mkdir -p "$LOG_ROOT" "$MPL_ROOT"
echo "Official constant-score evaluation: $EXPERIMENT_NAME"
echo "Prediction root: $PRED_ROOT"
echo "Every loaded prediction receives confidence 1.0 inside eval_scannet.py"
echo "Evaluator SHA256: $evaluator_sha"

(
    cd "$EVAL_CODE"
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR="$MPL_ROOT" \
    "$PYTHON" eval_scannet.py \
        --dataset scannet \
        --data_path /extra/ZhaoX/scannet_data/scans \
        --dump_dir eval_scannet \
        --num_point 40000 \
        --cluster_sampling seed_fps \
        --use_3d_nms \
        --use_cls_nms \
        --per_class_proposal \
        --gpu 0 \
        --pred_root "$PRED_ROOT"
) | tee "$LOG_PATH"

echo "Saved: $LOG_PATH"
