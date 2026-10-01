#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
PYTHON=/home/admin1/miniconda3/envs/boxfusion2/bin/python
GPU="${1:-0}"
SCENES="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
RUN_ROOT="$ROOT/results/scannet_m2nl_m5_dual_full100"
PERSISTENT="$RUN_ROOT/persistent"
CURRENT="$RUN_ROOT/current"
LOG_ROOT="$ROOT/logs/scannet_m2nl_m5_dual_full100"

mkdir -p "$PERSISTENT" "$CURRENT" "$LOG_ROOT"
cd "$ROOT"

sha256sum tools/integrated_online.py "$SCENES" > "$LOG_ROOT/input_sha256.txt"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
HF_HUB_OFFLINE=1 M2_MODE=nativelogit M2_EXCLUSIVE=1 \
M5_OFF=0 M5_CH2_OFF=0 CAUSAL_TAU=0.5 M5_SCORE_THR=1.0 \
CAUSAL_NAT="$ROOT/results/scannet_t05_boxer_kfmap_score05" \
CAUSAL_KFD="$ROOT/diagnostics/kfmap_score05" \
CAUSAL_OUT="$PERSISTENT" \
"$PYTHON" tools/integrated_online.py \
  --batch "$SCENES" \
  --score-view persistent \
  --current-out-root "$CURRENT" \
  2>&1 | tee "$LOG_ROOT/generate.log"

expected="$(awk 'NF && $1 !~ /^#/ {n += 1} END {print n + 0}' "$SCENES")"
for view in "$PERSISTENT" "$CURRENT"; do
  found="$(find "$view" -maxdepth 1 -type f -name 'scene*_boxes.pkl' | wc -l)"
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

bash scripts/eval_scannet_official100_real_score.sh \
  scannet_m2nl_m5_dual_persistent "$PERSISTENT" \
  2>&1 | tee "$LOG_ROOT/eval_persistent.log"
bash scripts/eval_scannet_official100_real_score.sh \
  scannet_m2nl_m5_dual_current "$CURRENT" \
  2>&1 | tee "$LOG_ROOT/eval_current.log"

echo "SCANNET_M2NL_M5_DUAL_FULL100_DONE"
