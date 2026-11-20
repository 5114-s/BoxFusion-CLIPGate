#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
REPORT="$ROOT/reports/original_boxfusion_vs_final_recar_efficiency100_20261005"
mkdir -p "$REPORT"
exec > >(tee -a "$REPORT/queue.log") 2>&1

echo "[$(date '+%F %T')] Waiting for recar_mvsr_design_100"
while tmux has-session -t recar_mvsr_design_100 2>/dev/null; do
  sleep 60
done

echo "[$(date '+%F %T')] MVSR task ended; waiting for both GPUs to be free"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | grep -q '[0-9]'; do
  sleep 60
done

echo "[$(date '+%F %T')] Starting matched efficiency official100"
exec bash "$ROOT/development/final_efficiency_original_vs_recar_20261005/run_official100.sh"
