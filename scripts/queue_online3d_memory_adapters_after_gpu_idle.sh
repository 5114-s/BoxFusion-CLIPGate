#!/usr/bin/env bash
set -euo pipefail

# Wait behind the jobs already using this machine, then evaluate the released
# Online3D Memory-based Adapters checkpoint on the frozen BoxFusion subsets.

ROOT=/data/ZhaoX/BoxFusion
ENV_ROOT="$ROOT/.conda/online3d_py38"
PYTHON="$ENV_ROOT/bin/python"
RUNNER="$ROOT/tools/run_online3d_memory_adapters_benchmark.py"
ONLINE3D_ROOT="$ROOT/external/Online3D_official_20261108"
CHECKPOINT="$ONLINE3D_ROOT/checkpoints/FCAF3D-SV+Ours.pth"
SCANNET_OUT="$ROOT/results/memory_adapters_scannet100"
CA1M_OUT="$ROOT/results/memory_adapters_ca1m107"
SMOKE_OUT="$ROOT/results/memory_adapters_smoke"
LOG_ROOT="$ROOT/logs/memory_adapters_benchmark"
STATUS_LOG="$LOG_ROOT/status.log"
MIN_FREE_MIB="${MIN_FREE_MIB:-22000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

RUNTIME_LIBS="/home/admin1/miniconda3/envs/cognav/lib:/home/admin1/miniconda3/envs/cognav/targets/x86_64-linux/lib:$ENV_ROOT/lib"

mkdir -p "$LOG_ROOT" "$LOG_ROOT/mplconfig" "$LOG_ROOT/cache" \
  "$SCANNET_OUT" "$CA1M_OUT" "$SMOKE_OUT"

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

for required in "$PYTHON" "$RUNNER" "$CHECKPOINT"; do
  [[ -e "$required" ]] || { status "ERROR missing required input: $required"; exit 1; }
done
command -v nvidia-smi >/dev/null || { status "ERROR nvidia-smi is unavailable"; exit 1; }

exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { status "ERROR Memory-based Adapters queue is already active"; exit 1; }

status "queued; waiting for both GPUs to have at least ${MIN_FREE_MIB} MiB free"
stable_polls=0
while (( stable_polls < 3 )); do
  mapfile -t free_mib < <(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
  )
  if (( ${#free_mib[@]} >= 2 )) \
    && (( free_mib[0] >= MIN_FREE_MIB )) \
    && (( free_mib[1] >= MIN_FREE_MIB )); then
    stable_polls=$((stable_polls + 1))
    status "GPU idle check ${stable_polls}/3 passed: free=${free_mib[*]} MiB"
  else
    stable_polls=0
    status "GPU jobs still active: free=${free_mib[*]:-unknown} MiB"
  fi
  if (( stable_polls < 3 )); then
    sleep "$POLL_SECONDS"
  fi
done

status "GPU gate passed; validating the compiled CUDA extensions"
CUDA_VISIBLE_DEVICES=1 run_python -c \
  'import torch, MinkowskiEngine, sort_vertices; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0), MinkowskiEngine.__version__)' \
  2>&1 | tee "$LOG_ROOT/cuda_runtime.log"

status "starting two-frame ScanNet smoke test on GPU 1"
CUDA_VISIBLE_DEVICES=1 run_python "$RUNNER" \
  --dataset scannet \
  --scene scene0568_00 \
  --output-root "$SMOKE_OUT" \
  --max-frames 2 \
  2>&1 | tee "$LOG_ROOT/smoke_scannet.log"
status "smoke test passed"

run_two_shards() {
  local dataset="$1"
  local output_root="$2"
  local dataset_log="$LOG_ROOT/$dataset"
  mkdir -p "$dataset_log"

  status "starting $dataset inference with one deterministic shard per GPU"
  CUDA_VISIBLE_DEVICES=0 run_python "$RUNNER" \
    --dataset "$dataset" \
    --output-root "$output_root" \
    --num-shards 2 \
    --shard-index 0 2>&1 | tee "$dataset_log/gpu0.log" &
  local pid0=$!

  CUDA_VISIBLE_DEVICES=1 run_python "$RUNNER" \
    --dataset "$dataset" \
    --output-root "$output_root" \
    --num-shards 2 \
    --shard-index 1 2>&1 | tee "$dataset_log/gpu1.log" &
  local pid1=$!

  local failed=0
  wait "$pid0" || failed=1
  wait "$pid1" || failed=1
  (( failed == 0 )) || { status "ERROR $dataset inference failed"; return 1; }
  status "$dataset inference completed"
}

run_two_shards scannet "$SCANNET_OUT"
scannet_count="$(find "$SCANNET_OUT" -maxdepth 1 -type f -name 'scene*_boxes.pkl' | wc -l)"
[[ "$scannet_count" -eq 100 ]] || {
  status "ERROR expected 100 ScanNet predictions, found $scannet_count"
  exit 1
}
status "running the frozen ScanNetV2-100 class-agnostic evaluator"
bash "$ROOT/scripts/eval_scannet_official100_real_score.sh" \
  memory_adapters_scannet100 "$SCANNET_OUT" \
  2>&1 | tee "$LOG_ROOT/eval_scannet100.log"

run_two_shards ca1m "$CA1M_OUT"
ca1m_count="$(find "$CA1M_OUT" -maxdepth 1 -type f -name '*_boxes.pkl' | wc -l)"
[[ "$ca1m_count" -eq 107 ]] || {
  status "ERROR expected 107 CA-1M predictions, found $ca1m_count"
  exit 1
}
status "running the frozen CA-1M-107 class-agnostic evaluator"
/home/admin1/miniconda3/envs/boxfusion2/bin/python "$ROOT/tools/eval_ca1m.py" \
  --pred-root "$CA1M_OUT" 2>&1 | tee "$LOG_ROOT/eval_ca1m107.log"

status "COMPLETE: Memory-based Adapters ScanNetV2-100 and CA-1M-107 results are ready"
