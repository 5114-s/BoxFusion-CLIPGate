#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
LOG_ROOT="$ROOT/logs/scannet_strict_causal_online_official100"
ARCHIVE="$ROOT/reports/strict_causal_online_official100_20260922/scene0304_00_pre_repair"
MAIN="$ROOT/scripts/run_scannet_strict_causal_online_official100.sh"

mkdir -p "$ARCHIVE"
exec > >(tee -a "$LOG_ROOT/finalizer.log") 2>&1
echo "[$(date '+%F %T')] finalizer waiting for the primary official100 lock"
exec 8>"$LOG_ROOT/run.lock"
flock 8
flock -u 8
exec 8>&-
echo "[$(date '+%F %T')] primary batch released its lock; repairing scene0304_00"

move_if_present() {
  local source="$1" group="$2"
  if [[ -e "$source" ]]; then
    mkdir -p "$ARCHIVE/$group"
    mv "$source" "$ARCHIVE/$group/"
  fi
}

move_if_present \
  "$ROOT/results/scannet_strict_causal_online_official100/scene0304_00_boxes.pkl" online
move_if_present \
  "$ROOT/results/scannet_strict_causal_online_official100_native/scene0304_00_boxes.pkl" native
move_if_present \
  "$ROOT/diagnostics/strict_causal_online_official100/scene0304_00.json" runtime
move_if_present \
  "$ROOT/diagnostics/strict_causal_online_official100_native/scene0304_00_boxer_lifting.jsonl" native_diagnostics
for name in scene0304_00_pvq_ar.jsonl scene0304_00_pvq_ar_summary.json \
  scene0304_00_pvq_kfmap.jsonl scene0304_00_pvq_nms.jsonl; do
  move_if_present \
    "$ROOT/diagnostics/strict_causal_online_official100_pvq/$name" pvq
done
move_if_present \
  "$LOG_ROOT/scenes/scene0304_00.log" log

echo "[$(date '+%F %T')] restarting resumable runner for repair and final summary"
bash "$MAIN"
printf '%s\n' \
  "Resolved: scene0304_00 was rerun after the primary background batch." \
  > "$ROOT/reports/strict_causal_online_official100_20260922/RERUN_RESOLVED.txt"
echo "[$(date '+%F %T')] finalizer complete"
