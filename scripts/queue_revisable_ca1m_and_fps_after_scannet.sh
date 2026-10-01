#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
QUEUE_LOG="$ROOT/logs/revisable_followup_queue_20260925.log"
SCANNET_SESSION=scannet_plr_revisable_v4_official100_20260925
SCANNET_REPORT="$ROOT/reports/strict_causal_nochild_revisable_official100_v4_20260925/REPORT.md"
CA_SESSION=ca1m_revisable_nochild_factorial107_v2_20260925
FPS_SESSION=bf_original_flow_revisable_fps100_v2_20260925

exec > >(tee -a "$QUEUE_LOG") 2>&1
echo "[$(date '+%F %T')] Waiting for revised ScanNet official100"
while tmux has-session -t "$SCANNET_SESSION" 2>/dev/null; do
  sleep 60
done

[[ -s "$SCANNET_REPORT" ]] || {
  echo "[$(date '+%F %T')] Revised ScanNet run ended without a final report" >&2
  exit 1
}

echo "[$(date '+%F %T')] Starting revised CA-1M on GPU1 and matched FPS on GPU0"
tmux new-session -d -s "$CA_SESSION" \
  "cd '$ROOT' && bash scripts/run_ca1m_strict_causal_nochild_revisable_factorial107_v2.sh"
tmux new-session -d -s "$FPS_SESSION" \
  "cd '$ROOT' && bash scripts/run_scannet_original_flow_revisable_fps100_v2.sh"

while tmux has-session -t "$CA_SESSION" 2>/dev/null \
   || tmux has-session -t "$FPS_SESSION" 2>/dev/null; do
  sleep 60
done

CA_REPORT="$ROOT/reports/ca1m_strict_causal_nochild_revisable_factorial_v2_20260925/REPORT.md"
FPS_REPORT="$ROOT/reports/original_flow_revisable_strict_causal_nochild_fps_official100_v2_20260925/REPORT.md"
[[ -s "$CA_REPORT" ]] || {
  echo "[$(date '+%F %T')] Revised CA-1M run did not produce REPORT.md" >&2
  exit 1
}
[[ -s "$FPS_REPORT" ]] || {
  echo "[$(date '+%F %T')] Revised FPS run did not produce REPORT.md" >&2
  exit 1
}
echo "[$(date '+%F %T')] Revised CA-1M and FPS experiments complete"
