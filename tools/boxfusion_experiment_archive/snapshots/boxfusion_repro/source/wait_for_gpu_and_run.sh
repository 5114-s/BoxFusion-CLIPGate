#!/usr/bin/env bash
set -euo pipefail

BLOCKING_PID="${1:-318655}"
HOST_GPU="${2:-0}"
WORK_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_repro
RUN_SCRIPT="$WORK_ROOT/run_scannet_sens_rgb_scorefix.sh"

echo "[$(date '+%F %T')] Waiting for existing training PID $BLOCKING_PID"
while ps -p "$BLOCKING_PID" >/dev/null 2>&1; do
    status=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits)
    echo "[$(date '+%F %T')] Existing training is active; GPU status: $status"
    sleep 60
done

echo "[$(date '+%F %T')] Training PID exited; waiting for GPU $HOST_GPU to be stably idle"
idle_checks=0
while (( idle_checks < 3 )); do
    read -r memory_used utilization < <(
        nvidia-smi --id="$HOST_GPU" \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits |
            tr -d ',' 
    )
    if (( memory_used < 500 && utilization < 10 )); then
        idle_checks=$((idle_checks + 1))
    else
        idle_checks=0
    fi
    echo "[$(date '+%F %T')] GPU $HOST_GPU: memory=${memory_used}MiB, util=${utilization}%, idle_checks=$idle_checks/3"
    if (( idle_checks < 3 )); then
        sleep 60
    fi
done

echo "[$(date '+%F %T')] GPU $HOST_GPU is idle; starting BoxFusion"
exec bash "$RUN_SCRIPT" "$HOST_GPU"
