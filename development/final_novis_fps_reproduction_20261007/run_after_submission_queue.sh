#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
ACTIVE_PID=392861
QUEUE_DONE="$ROOT/reports/submission_completion_queue_20261006/DONE"
OUTPUT="$ROOT/reports/final_novis_recar_fps_reproduction100_20261007"
LOG="$OUTPUT/driver.log"
PYTHON=/home/admin1/miniconda3/envs/boxfusion-online/bin/python

mkdir -p "$OUTPUT"
printf '%s\twaiting_for_submission_queue_pid_%s\n' "$(date --iso-8601=seconds)" "$ACTIVE_PID" | tee -a "$LOG"
while kill -0 "$ACTIVE_PID" 2>/dev/null; do
  sleep 60
done
while [[ ! -f "$QUEUE_DONE" ]]; do
  printf '%s\twaiting_for_submission_queue_DONE\n' "$(date --iso-8601=seconds)" | tee -a "$LOG"
  sleep 300
done

printf '%s\tstarting_final_novis_recar_scannet100\n' "$(date --iso-8601=seconds)" | tee -a "$LOG"
cd "$ROOT"
"$PYTHON" \
  development/final_novis_fps_reproduction_20261007/run_latest_novis_only100.py \
  --output "$OUTPUT" \
  --device-index 0 \
  --limit 100 2>&1 | tee -a "$LOG"
printf '%s\tfinished_final_novis_recar_scannet100\n' "$(date --iso-8601=seconds)" | tee -a "$LOG"
