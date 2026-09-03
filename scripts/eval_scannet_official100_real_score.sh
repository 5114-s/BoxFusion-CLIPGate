#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 EXPERIMENT_NAME [PREDICTION_ROOT]" >&2
  exit 2
fi

EXPERIMENT="$1"
[[ "$EXPERIMENT" =~ ^[A-Za-z0-9._-]+$ ]] || \
  { echo "Invalid experiment name: $EXPERIMENT" >&2; exit 2; }

ROOT=/data/ZhaoX/BoxFusion
PYTHON=/home/admin1/miniconda3/envs/boxfusion2/bin/python
EVAL_ROOT="$ROOT/evaluation"
EVALUATOR="$EVAL_ROOT/eval_scannet.py"
EXPECTED_EVALUATOR_SHA=7f32a0c8120d1233e7393909b2f1d4a526ed4a23d8d94b535dd7423eae41f8df
PRED_ROOT="${2:-$ROOT/results/$EXPERIMENT}"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
LOG_ROOT="$ROOT/logs/scannet_official100_real_score"
LOG_PATH="$LOG_ROOT/${EXPERIMENT}.log"
MPL_ROOT="$LOG_ROOT/mplconfig"

[[ -d "$PRED_ROOT" ]] || { echo "Missing prediction root: $PRED_ROOT" >&2; exit 1; }
PRED_ROOT="$(realpath "$PRED_ROOT")"
[[ "$(sha256sum "$EVALUATOR" | awk '{print $1}')" == "$EXPECTED_EVALUATOR_SHA" ]] || \
  { echo "Real-score evaluator hash mismatch" >&2; exit 1; }
[[ "$(sha256sum "$SCENE_LIST" | awk '{print $1}')" == \
  4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5 ]] || \
  { echo "Official scene-list hash mismatch" >&2; exit 1; }

expected="$(awk 'NF && $1 !~ /^#/ {n += 1} END {print n + 0}' "$SCENE_LIST")"
found="$(find "$PRED_ROOT" -maxdepth 1 -type f -name 'scene*_boxes.pkl' | wc -l)"
[[ "$found" -eq "$expected" ]] || \
  { echo "Expected $expected predictions, found $found" >&2; exit 1; }
while IFS= read -r scene || [[ -n "$scene" ]]; do
  [[ -z "$scene" || "$scene" == \#* ]] && continue
  [[ -s "$PRED_ROOT/${scene}_boxes.pkl" ]] || \
    { echo "Missing prediction: $scene" >&2; exit 1; }
done < "$SCENE_LIST"

mkdir -p "$LOG_ROOT" "$MPL_ROOT"
echo "Official100 real-score evaluation: $EXPERIMENT"
echo "Prediction root: $PRED_ROOT"
echo "Evaluator SHA256: $EXPECTED_EVALUATOR_SHA"
(
  cd "$EVAL_ROOT"
  CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$MPL_ROOT" \
  "$PYTHON" eval_scannet.py \
    --dataset scannet \
    --data_path /extra/ZhaoX/scannet_data/scans \
    --num_point 40000 \
    --cluster_sampling seed_fps \
    --use_3d_nms \
    --use_cls_nms \
    --per_class_proposal \
    --num_workers 0 \
    --gpu 0 \
    --pred_root "$PRED_ROOT"
) | tee "$LOG_PATH"
echo "Saved: $LOG_PATH"
