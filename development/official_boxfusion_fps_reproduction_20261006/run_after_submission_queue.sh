#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
ACTIVE_PID=321040
OUTPUT="$ROOT/reports/original_boxfusion_fps_reproduction100_20261006"
LOG="$OUTPUT/driver.log"
PYTHON=/home/admin1/miniconda3/envs/boxfusion-online/bin/python

mkdir -p "$OUTPUT"
printf '%s\twaiting_for_submission_queue_pid_%s\n' "$(date --iso-8601=seconds)" "$ACTIVE_PID" | tee -a "$LOG"
while kill -0 "$ACTIVE_PID" 2>/dev/null; do
  sleep 60
done

printf '%s\tstarting_original_boxfusion_only_scannet100\n' "$(date --iso-8601=seconds)" | tee -a "$LOG"
cd "$ROOT"
"$PYTHON" \
  development/official_boxfusion_fps_reproduction_20261006/run_official_only100.py \
  --output "$OUTPUT" \
  --device-index 0 \
  --limit 100 2>&1 | tee -a "$LOG"
printf '%s\tfinished_original_boxfusion_only_scannet100\n' "$(date --iso-8601=seconds)" | tee -a "$LOG"
