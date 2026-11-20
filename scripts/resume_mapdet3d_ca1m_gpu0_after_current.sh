#!/usr/bin/env bash
set -euo pipefail

# Finish the interrupted Map-Det3D CA-1M-107 benchmark on physical GPU 0.
# Existing, readable scene predictions are reused by the Python runner.

ROOT=/data/ZhaoX/BoxFusion
PYTHON="$ROOT/.conda/mapdet3d_py311/bin/python"
RUNNER="$ROOT/tools/run_mapdet3d_scene_benchmark.py"
MODEL_CACHE="$ROOT/models/mapdet3d_hf_cache"
CA1M_OUT="$ROOT/results/mapdet3d_ca1m107"
LOG_ROOT="$ROOT/logs/mapdet3d_benchmark/resume_gpu0"
STATUS_LOG="$LOG_ROOT/status.log"
MIN_FREE_MIB="${MIN_FREE_MIB:-22000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$LOG_ROOT" "$CA1M_OUT"

status() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$STATUS_LOG"
}

for required in "$PYTHON" "$RUNNER"; do
  [[ -e "$required" ]] || { status "ERROR missing required input: $required"; exit 1; }
done
command -v nvidia-smi >/dev/null || { status "ERROR nvidia-smi is unavailable"; exit 1; }

exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { status "ERROR GPU0 resume queue is already active"; exit 1; }

initial_count="$(find "$CA1M_OUT" -maxdepth 1 -type f -name '*_boxes.pkl' | wc -l)"
status "queued with ${initial_count}/107 outputs; waiting for physical GPU 0 to become idle"

# The current Map-Det3D shard keeps GPU 0 below this threshold. Requiring three
# consecutive checks prevents this queue from racing a short allocation gap.
stable_polls=0
while (( stable_polls < 3 )); do
  free_mib="$(
    nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits \
      | tr -d ' '
  )"
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
    stable_polls=$((stable_polls + 1))
    status "GPU0 idle check ${stable_polls}/3 passed: free=${free_mib} MiB"
  else
    stable_polls=0
    current_count="$(find "$CA1M_OUT" -maxdepth 1 -type f -name '*_boxes.pkl' | wc -l)"
    status "GPU0 still active: free=${free_mib:-unknown} MiB, outputs=${current_count}/107"
  fi
  if (( stable_polls < 3 )); then
    sleep "$POLL_SECONDS"
  fi
done

status "GPU0 gate passed; validating CUDA runtime"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))' \
  2>&1 | tee "$LOG_ROOT/cuda_runtime.log"

status "resuming every missing CA-1M scene on physical GPU 0"
CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" "$RUNNER" \
  --dataset ca1m \
  --output-root "$CA1M_OUT" \
  --model-cache "$MODEL_CACHE" \
  2>&1 | tee "$LOG_ROOT/inference.log"

ca1m_count="$(find "$CA1M_OUT" -maxdepth 1 -type f -name '*_boxes.pkl' | wc -l)"
[[ "$ca1m_count" -eq 107 ]] || {
  status "ERROR expected 107 CA-1M predictions, found $ca1m_count"
  exit 1
}

status "running the frozen CA-1M-107 class-agnostic evaluator"
/home/admin1/miniconda3/envs/boxfusion2/bin/python "$ROOT/tools/eval_ca1m.py" \
  --pred-root "$CA1M_OUT" 2>&1 | tee "$LOG_ROOT/eval_ca1m107.log"

status "COMPLETE: Map-Det3D CA-1M-107 results are ready"
