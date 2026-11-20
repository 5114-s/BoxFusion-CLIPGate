#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
DEV="$ROOT/development/final_efficiency_original_vs_recar_20261005"
GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', *' '$1 == 0 {print $2}')"

while tmux has-session -t recar_crossdataset_novis 2>/dev/null \
   || tmux has-session -t recar_3rscan_novis_queue 2>/dev/null; do
  sleep 60
done

while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
    | grep -Fxq "$GPU_UUID"; do
  sleep 60
done

exec bash "$DEV/run_official100_prewarm.sh"
