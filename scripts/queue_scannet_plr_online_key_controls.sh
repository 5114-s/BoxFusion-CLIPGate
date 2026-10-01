#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
LOG_ROOT="$ROOT/logs/scannet_plr_online_key_controls_20260925"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/queue.log") 2>&1

echo "[$(date '+%F %T')] Waiting for current GPU experiments"
while tmux has-session -t ca1m_strict_nochild_factorial107_20260925 2>/dev/null \
   || tmux has-session -t bf_original_flow_fps100_20260925 2>/dev/null; do
  sleep 60
done
[[ -s "$ROOT/reports/ca1m_strict_causal_nochild_factorial_20260925/REPORT.md" ]] || {
  echo "[$(date '+%F %T')] CA-1M experiment did not produce its final report; refusing to start" >&2
  exit 1
}
echo "[$(date '+%F %T')] Current experiments finished; starting PLR controls"
exec bash "$ROOT/scripts/run_scannet_plr_online_key_controls.sh"
