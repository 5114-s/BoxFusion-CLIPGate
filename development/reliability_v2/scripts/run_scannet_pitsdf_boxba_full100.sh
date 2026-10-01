#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
EXPERIMENT=scannet_cbest_real_score_pitsdf_boxba_score05
OUTPUT_ROOT="$ROOT/results/$EXPERIMENT"
LOG_ROOT="$ROOT/logs/$EXPERIMENT"
MATERIALIZER="$ROOT/tools/materialize_scannet_f4_stream3dv2_lite_full100.py"
EVAL_RUNNER="$ROOT/scripts/eval_scannet_official100_real_score.sh"

for required in "$PYTHON" "$MATERIALIZER" "$EVAL_RUNNER"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done
[[ ! -e "$OUTPUT_ROOT" ]] || \
  { echo "Refusing to overwrite output root: $OUTPUT_ROOT" >&2; exit 1; }

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1
echo "[$(date '+%F %T')] bounded PI-TSDF + free-space + 7D BoxBA official100 started"
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$MATERIALIZER" \
  --pitsdf-boxba \
  --output-root "$OUTPUT_ROOT"
bash "$EVAL_RUNNER" "$EXPERIMENT" "$OUTPUT_ROOT"
echo "[$(date '+%F %T')] bounded PI-TSDF + free-space + 7D BoxBA official100 complete"
