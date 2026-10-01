#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
EXPERIMENT=scannet_cbest_real_score_r15_ovir_openbox_mhobb_lowappend_score05
OUTPUT_ROOT="$ROOT/results/$EXPERIMENT"
LOG_ROOT="$ROOT/logs/$EXPERIMENT"
MATERIALIZER="$ROOT/tools/materialize_scannet_r15_ovir_openbox_mhobb_full100.py"
EVAL_RUNNER="$ROOT/scripts/eval_scannet_official100_real_score.sh"
R15_SIDECAR="$ROOT/logs/scannet_target_first_mobilesam_masklift_full100_score05/TARGET_FIRST_MOBILESAM_MASKLIFT_FULL100.json"
RGBD_ROOT="$ROOT/upstream_clean/scannet_readme_frames"

for required in \
  "$PYTHON" "$MATERIALIZER" "$EVAL_RUNNER" "$R15_SIDECAR" "$RGBD_ROOT"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done
[[ ! -e "$OUTPUT_ROOT" ]] || \
  { echo "Refusing to overwrite output root: $OUTPUT_ROOT" >&2; exit 1; }

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1
echo "[$(date '+%F %T')] R15 OVIR/OpenBox/MH-OBB official100 started"
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$MATERIALIZER" \
  --r15-sidecar "$R15_SIDECAR" \
  --scene-root "$RGBD_ROOT" \
  --output-root "$OUTPUT_ROOT"
bash "$EVAL_RUNNER" "$EXPERIMENT" "$OUTPUT_ROOT"
echo "[$(date '+%F %T')] R15 OVIR/OpenBox/MH-OBB official100 complete"
