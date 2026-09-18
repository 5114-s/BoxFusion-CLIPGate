#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
PYTHON=/home/admin1/miniconda3/envs/boxfusion2/bin/python
GPU="${1:-1}"
SCENES="$ROOT/tools/boxfusion_tr3d_pipeline/evaluation/data_util/meta_data/ca1m_val_full107.txt"
RUN_ROOT="$ROOT/results/ca1m_m2nl_m5_dual_full107"
PERSISTENT="$RUN_ROOT/persistent"
CURRENT="$RUN_ROOT/current"
LOG_ROOT="$ROOT/logs/ca1m_m2nl_m5_dual_full107"
EVAL_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/evaluation

mkdir -p "$PERSISTENT" "$CURRENT" "$LOG_ROOT"
cd "$ROOT"

sha256sum tools/integrated_online.py "$SCENES" > "$LOG_ROOT/input_sha256.txt"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
HF_HUB_OFFLINE=1 M2_MODE=nativelogit M2_EXCLUSIVE=1 \
M5_OFF=0 M5_CH2_OFF=0 CAUSAL_TAU=0.5 M5_SCORE_THR=1.0 \
CAUSAL_NAT="$ROOT/results/ca1m_prod" \
CAUSAL_KFD="$ROOT/diagnostics/ca1m_prod" \
CAUSAL_OUT="$PERSISTENT" \
"$PYTHON" tools/integrated_online.py \
  --batch "$SCENES" \
  --score-view persistent \
  --current-out-root "$CURRENT" \
  2>&1 | tee "$LOG_ROOT/generate.log"

expected="$(awk 'NF && $1 !~ /^#/ {n += 1} END {print n + 0}' "$SCENES")"
for view in "$PERSISTENT" "$CURRENT"; do
  found="$(find "$view" -maxdepth 1 -type f -name '*_boxes.pkl' | wc -l)"
  [[ "$found" -eq "$expected" ]] || {
    echo "Coverage failure: $view has $found/$expected prediction files" >&2
    exit 1
  }
  while IFS= read -r scene || [[ -n "$scene" ]]; do
    [[ -z "$scene" || "$scene" == \#* ]] && continue
    [[ -s "$view/${scene}_boxes.pkl" ]] || {
      echo "Missing prediction: $view/${scene}_boxes.pkl" >&2
      exit 1
    }
    [[ -s "$view/${scene}_boxes.pkl.dual_state.json" ]] || {
      echo "Missing dual-state sidecar: $view/${scene}_boxes.pkl.dual_state.json" >&2
      exit 1
    }
  done < "$SCENES"
done

cd "$EVAL_ROOT"
CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR=/tmp/mpl_ca1m_m2nl_m5_dual_persistent \
"$PYTHON" eval_ca1m.py \
  --dataset ca1m \
  --data_path /tmp/ca1m_clean_root \
  --pred_root "$PERSISTENT" \
  --use_3d_nms --per_class_proposal \
  2>&1 | tee "$LOG_ROOT/eval_persistent.log"
CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR=/tmp/mpl_ca1m_m2nl_m5_dual_current \
"$PYTHON" eval_ca1m.py \
  --dataset ca1m \
  --data_path /tmp/ca1m_clean_root \
  --pred_root "$CURRENT" \
  --use_3d_nms --per_class_proposal \
  2>&1 | tee "$LOG_ROOT/eval_current.log"

echo "CA1M_M2NL_M5_DUAL_FULL107_DONE"
