#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
EXPERIMENT=scannet_cbest_real_score_oas_p1_voxelhash_birth_score05
OUTPUT_ROOT="$ROOT/results/$EXPERIMENT"
LOG_ROOT="$ROOT/logs/$EXPERIMENT"
MATERIALIZER="$ROOT/tools/materialize_scannet_oas_p1_voxelhash_birth_full100.py"
EVAL_RUNNER="$ROOT/scripts/eval_scannet_official100_real_score.sh"
F2_ROOT="$ROOT/logs/scannet_fastsam_f2_paper100_score05"
NATIVE_ROOT="$ROOT/results/scannet_t05_boxer_replay_active_score05"

for required in \
  "$PYTHON" "$MATERIALIZER" "$EVAL_RUNNER" "$F2_ROOT" "$NATIVE_ROOT"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done
[[ ! -e "$OUTPUT_ROOT" ]] || \
  { echo "Refusing to overwrite output root: $OUTPUT_ROOT" >&2; exit 1; }

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1
echo "[$(date '+%F %T')] OAS-P1 voxel-hash birth official100 started"
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$MATERIALIZER" \
  --f2-root "$F2_ROOT" \
  --native-root "$NATIVE_ROOT" \
  --output-root "$OUTPUT_ROOT"
bash "$EVAL_RUNNER" "$EXPERIMENT" "$OUTPUT_ROOT"
echo "[$(date '+%F %T')] OAS-P1 voxel-hash birth official100 complete"
