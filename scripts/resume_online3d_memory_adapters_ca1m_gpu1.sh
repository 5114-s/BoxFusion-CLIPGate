#!/usr/bin/env bash
set -euo pipefail

# Resume only the unfinished CA-1M evaluation on physical GPU 1. Existing
# atomic per-scene predictions are validated and reused by the Python runner.

ROOT=/data/ZhaoX/BoxFusion
ENV_ROOT="$ROOT/.conda/online3d_py38"
PYTHON="$ENV_ROOT/bin/python"
RUNNER="$ROOT/tools/run_online3d_memory_adapters_benchmark.py"
ONLINE3D_ROOT="$ROOT/external/Online3D_official_20261108"
OUTPUT_ROOT="$ROOT/results/memory_adapters_ca1m107"
LOG_ROOT="$ROOT/logs/memory_adapters_benchmark/gpu1"
STATUS_LOG="$LOG_ROOT/status.log"
MIN_FREE_MIB="${MIN_FREE_MIB:-22000}"
POLL_SECONDS="${POLL_SECONDS:-5}"
RUNTIME_LIBS="/home/admin1/miniconda3/envs/cognav/lib:/home/admin1/miniconda3/envs/cognav/targets/x86_64-linux/lib:$ENV_ROOT/lib"

mkdir -p "$LOG_ROOT" "$LOG_ROOT/mplconfig" "$LOG_ROOT/cache" "$OUTPUT_ROOT"

status() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$STATUS_LOG"
}

run_python() {
  MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
  XDG_CACHE_HOME="$LOG_ROOT/cache" \
  PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=8 \
  LD_LIBRARY_PATH="$RUNTIME_LIBS" \
  PYTHONPATH="$ONLINE3D_ROOT:$ROOT" \
  "$PYTHON" "$@"
}

exec 9>"$ROOT/logs/memory_adapters_benchmark/run.lock"
flock -n 9 || { status "ERROR another Memory-based Adapters run is active"; exit 1; }

status "waiting to resume Memory-based Adapters CA-1M on physical GPU 1"
stable_polls=0
while (( stable_polls < 3 )); do
  free_mib="$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
    stable_polls=$((stable_polls + 1))
    status "GPU1 idle check ${stable_polls}/3 passed: free=${free_mib} MiB"
  else
    stable_polls=0
    status "GPU1 is busy: free=${free_mib:-unknown} MiB"
  fi
  if (( stable_polls < 3 )); then
    sleep "$POLL_SECONDS"
  fi
done

status "resuming CA-1M-107 inference; completed scene files will be reused"
CUDA_VISIBLE_DEVICES=1 run_python "$RUNNER" \
  --dataset ca1m \
  --output-root "$OUTPUT_ROOT" \
  2>&1 | tee -a "$LOG_ROOT/ca1m107.log"

scene_count="$(find "$OUTPUT_ROOT" -maxdepth 1 -type f -name '*_boxes.pkl' | wc -l)"
[[ "$scene_count" -eq 107 ]] || {
  status "ERROR expected 107 CA-1M predictions, found $scene_count"
  exit 1
}

status "running the frozen CA-1M-107 class-agnostic evaluator"
/home/admin1/miniconda3/envs/boxfusion2/bin/python "$ROOT/tools/eval_ca1m.py" \
  --pred-root "$OUTPUT_ROOT" 2>&1 | tee "$LOG_ROOT/eval_ca1m107.log"

status "COMPLETE: Memory-based Adapters CA-1M-107 results are ready"
