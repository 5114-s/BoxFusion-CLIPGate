#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
PYTHON=/home/admin1/miniconda3/envs/boxfusion2/bin/python
GPU="${1:-0}"
SCENES="$ROOT/tools/boxfusion_tr3d_pipeline/evaluation/data_util/meta_data/ca1m_val_full107.txt"
BASE="$ROOT/results/ca1m_thr15"
M1="$ROOT/results/ca1m_thr15_m1_full107"
M2="$ROOT/results/ca1m_thr15_m1_m2nl_full107"
KFD="$ROOT/diagnostics/ca1m_thr15"
LOG_ROOT="$ROOT/logs/ca1m_thr15_m1_m2nl_full107"
EVAL_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/evaluation

expected="$(awk 'NF && $1 !~ /^#/ {n += 1} END {print n + 0}' "$SCENES")"
base_found="$(find "$BASE" -maxdepth 1 -type f -name '*_boxes.pkl' | wc -l)"
[[ "$base_found" -eq "$expected" ]] || {
  echo "Baseline coverage failure: $BASE has $base_found/$expected prediction files" >&2
  exit 1
}
for output in "$M1" "$M2"; do
  if [[ -d "$output" ]] && find "$output" -maxdepth 1 -type f -name '*_boxes.pkl' -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty output: $output" >&2
    exit 1
  fi
done

mkdir -p "$M1" "$M2" "$KFD" "$LOG_ROOT"
cd "$ROOT"
sha256sum \
  tools/integrated_online.py \
  tools/verify_ca1m_m1_m2nl_pair.py \
  config/ca1m_thr15.yaml \
  "$SCENES" > "$LOG_ROOT/input_sha256.txt"
printf '%s\n' \
  'baseline=threshold0.15+TopK3+Boxer-active+real-score' \
  'M1=live WeDetect residual funnel (no NMS-child logs available for this frozen CA-1M baseline)' \
  'M2=nativelogit, exclusive support, tau=0.5, native rows only' \
  'M5=off' \
  'pairing=one model forward; M1 materialized before M2 score transform' \
  > "$LOG_ROOT/protocol.txt"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
HF_HUB_OFFLINE=1 M2_MODE=nativelogit M2_EXCLUSIVE=1 \
M5_OFF=1 M5_CH2_OFF=1 CAUSAL_TAU=0.5 FIXED_PRICE=0 \
CAUSAL_NAT="$BASE" CAUSAL_KFD="$KFD" CAUSAL_OUT="$M2" \
CAUSAL_M1_OUT="$M1" \
"$PYTHON" tools/integrated_online.py \
  --batch "$SCENES" \
  --score-view persistent \
  --m1-out-root "$M1" \
  2>&1 | tee "$LOG_ROOT/generate.log"

for output in "$M1" "$M2"; do
  found="$(find "$output" -maxdepth 1 -type f -name '*_boxes.pkl' | wc -l)"
  [[ "$found" -eq "$expected" ]] || {
    echo "Coverage failure: $output has $found/$expected prediction files" >&2
    exit 1
  }
  while IFS= read -r scene || [[ -n "$scene" ]]; do
    [[ -z "$scene" || "$scene" == \#* ]] && continue
    [[ -s "$output/${scene}_boxes.pkl" ]] || {
      echo "Missing prediction: $output/${scene}_boxes.pkl" >&2
      exit 1
    }
  done < "$SCENES"
done

"$PYTHON" tools/verify_ca1m_m1_m2nl_pair.py \
  --scenes "$SCENES" \
  --base-root "$BASE" \
  --m1-root "$M1" \
  --m2-root "$M2" \
  --out "$LOG_ROOT/pair_audit.json" \
  2>&1 | tee "$LOG_ROOT/pair_audit.log"

cd "$EVAL_ROOT"
for name in base_thr15 m1 m1_m2nl; do
  case "$name" in
    base_thr15) pred_root="$BASE" ;;
    m1) pred_root="$M1" ;;
    m1_m2nl) pred_root="$M2" ;;
  esac
  CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR="/tmp/mpl_ca1m_thr15_${name}" \
  "$PYTHON" eval_ca1m.py \
    --dataset ca1m \
    --data_path /tmp/ca1m_clean_root \
    --pred_root "$pred_root" \
    --use_3d_nms --per_class_proposal \
    2>&1 | tee "$LOG_ROOT/eval_${name}.log"
done

echo "CA1M_THR15_M1_M2NL_FULL107_DONE"
