#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/ZhaoX/BoxFusion
DEV="$MAIN/development/reliability_v2"
LOG="$MAIN/logs/reliability_v2_screening_queue_20260925.log"
BASE_SESSION=scannet_plr_revisable_v4_official100_20260925
BASE_REPORT="$MAIN/reports/strict_causal_nochild_revisable_official100_v4_20260925/REPORT.md"
SCREEN_SESSION=scannet_reliability_v2_screening_official100_20260925

exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%F %T')] Waiting for frozen v4 official100"
while tmux has-session -t "$BASE_SESSION" 2>/dev/null; do sleep 60; done
[[ -s "$BASE_REPORT" ]] || {
  echo "[$(date '+%F %T')] v4 ended without REPORT.md; screening not started" >&2
  exit 1
}
echo "[$(date '+%F %T')] Starting isolated reliability-v2 screening"
tmux new-session -d -s "$SCREEN_SESSION" \
  "cd '$DEV' && bash scripts/run_reliability_v2_screening_official100.sh"
while tmux has-session -t "$SCREEN_SESSION" 2>/dev/null; do sleep 60; done
REPORT="$DEV/reports/scannet_reliability_v2_screening_official100_20260925/REPORT.md"
[[ -s "$REPORT" ]] || {
  echo "[$(date '+%F %T')] screening ended without report" >&2; exit 1;
}
echo "[$(date '+%F %T')] Screening finished; CA-1M/FPS/final-paper jobs remain paused"
