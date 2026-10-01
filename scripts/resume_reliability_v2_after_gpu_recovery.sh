#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/ZhaoX/BoxFusion
DEV="$MAIN/development/reliability_v2"
LOG="$MAIN/logs/reliability_v2_gpu_recovery_queue_20260925.log"
RUN_LOG="$DEV/logs/scannet_reliability_v2_screening_official100"

exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%F %T')] Waiting for NVIDIA driver and two idle GPUs"

while true; do
  if values="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
    if awk -F, '
      BEGIN { ok=1; n=0 }
      { gsub(/ /, "", $1); gsub(/ /, "", $2); n++; if ($1 > 1500 || $2 > 10) ok=0 }
      END { exit !(n >= 2 && ok) }
    ' <<<"$values"; then
      break
    fi
  fi
  sleep 60
done

echo "[$(date '+%F %T')] GPU driver recovered and GPUs are idle"

# The interrupted process produced no accepted prediction or diagnostic file.
# Preserve its logs before starting a clean first-scene retry.
stamp="$(date '+%Y%m%d_%H%M%S')"
archive="$RUN_LOG/failed_attempts/${stamp}_gpu_driver_loss"
mkdir -p "$archive"
[[ ! -e "$RUN_LOG/driver.log" ]] || mv "$RUN_LOG/driver.log" "$archive/driver.log"
[[ ! -e "$RUN_LOG/scenes/scene0568_00.log" ]] || \
  mv "$RUN_LOG/scenes/scene0568_00.log" "$archive/scene0568_00.log"

for root in \
  "$DEV/results/scannet_reliability_v2_official100_native" \
  "$DEV/results/scannet_reliability_v2_official100" \
  "$DEV/diagnostics/scannet_reliability_v2_official100" \
  "$DEV/results/scannet_reliability_v2_components"; do
  if find "$root" -type f -print -quit 2>/dev/null | grep -q .; then
    echo "Unexpected partial artifact under $root; refusing automatic retry" >&2
    exit 1
  fi
done

echo "[$(date '+%F %T')] Restarting reliability-v2 screening from scene 1/100"
cd "$DEV"
exec bash scripts/run_reliability_v2_screening_official100.sh
