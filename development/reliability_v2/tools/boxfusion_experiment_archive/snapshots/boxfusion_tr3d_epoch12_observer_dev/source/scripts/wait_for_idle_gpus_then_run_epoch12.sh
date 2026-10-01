#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
GPU_SPEC="${1:-0,1}"
RUN_TAG="${2:-tr3d_t1_epoch12_fp32_observer10_v1}"
WAIT_PID="${BOXFUSION_TR3D_WAIT_PID:-}"
POLL_SECONDS="${BOXFUSION_TR3D_POLL_SECONDS:-30}"
IDLE_POLLS_REQUIRED="${BOXFUSION_TR3D_IDLE_POLLS:-3}"

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid poll interval" >&2; exit 2; }
[[ "$IDLE_POLLS_REQUIRED" =~ ^[1-9][0-9]*$ ]] || { echo "invalid idle poll count" >&2; exit 2; }

gpu_compute_pids() {
  local gpu="$1"
  local output
  if ! output="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>&1)"; then
    echo "nvidia-smi query failed for GPU $gpu: $output" >&2
    return 2
  fi
  printf '%s\n' "$output" | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/, ""); print}'
}

if [[ -n "$WAIT_PID" ]]; then
  [[ "$WAIT_PID" =~ ^[1-9][0-9]*$ ]] || { echo "invalid wait PID" >&2; exit 2; }
  echo "Waiting for protected job PID $WAIT_PID to exit"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
fi

IFS=',' read -r -a requested_gpus <<< "$GPU_SPEC"
idle_polls=0
while (( idle_polls < IDLE_POLLS_REQUIRED )); do
  busy=0
  for gpu in "${requested_gpus[@]}"; do
    if ! pids="$(gpu_compute_pids "$gpu")"; then
      busy=1
      continue
    fi
    if [[ -n "$pids" ]]; then
      busy=1
      echo "GPU $gpu still busy: $(echo "$pids" | tr '\n' ' ')"
    fi
  done
  if [[ "$busy" == "0" ]]; then
    idle_polls=$((idle_polls + 1))
    echo "Idle GPU poll $idle_polls/$IDLE_POLLS_REQUIRED"
  else
    idle_polls=0
  fi
  if (( idle_polls < IDLE_POLLS_REQUIRED )); then
    sleep "$POLL_SECONDS"
  fi
done

echo "Requested GPUs stayed idle; starting isolated epoch12 observer"
exec bash "$ROOT/scripts/run_epoch12_observer_experiment.sh" \
  "$GPU_SPEC" "$RUN_TAG"
