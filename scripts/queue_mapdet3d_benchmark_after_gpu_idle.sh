#!/usr/bin/env bash
set -euo pipefail

# Queue the released Map-Det3D checkpoint behind the jobs that are already
# occupying this machine.  No evaluation scene is used for training.

ROOT=/data/ZhaoX/BoxFusion
PYTHON="$ROOT/.conda/mapdet3d_py311/bin/python"
RUNNER="$ROOT/tools/run_mapdet3d_scene_benchmark.py"
MODEL_CACHE="$ROOT/models/mapdet3d_hf_cache"
CUDA_OPS_ROOT="$ROOT/external/vis4d_cuda_ops_official_20261108"
SCANNET_OUT="$ROOT/results/mapdet3d_scannet100"
CA1M_OUT="$ROOT/results/mapdet3d_ca1m107"
SMOKE_OUT="$ROOT/results/mapdet3d_smoke"
LOG_ROOT="$ROOT/logs/mapdet3d_benchmark"
STATUS_LOG="$LOG_ROOT/status.log"
MIN_FREE_MIB="${MIN_FREE_MIB:-22000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$LOG_ROOT" "$MODEL_CACHE"

status() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$STATUS_LOG"
}

[[ -x "$PYTHON" ]] || { status "ERROR missing environment: $PYTHON"; exit 1; }
[[ -f "$RUNNER" ]] || { status "ERROR missing runner: $RUNNER"; exit 1; }
command -v nvidia-smi >/dev/null || { status "ERROR nvidia-smi is unavailable"; exit 1; }

status "queued; waiting for two GPUs to each have at least ${MIN_FREE_MIB} MiB free"
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

status "GPU gate passed; validating CUDA runtime"
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))' \
  2>&1 | tee "$LOG_ROOT/cuda_runtime.log"

if ! CUDA_VISIBLE_DEVICES=1 "$PYTHON" -c 'import torch; import vis4d_cuda_ops' >/dev/null 2>&1; then
  status "building the official vis4d CUDA operators for RTX 3090 (sm_86)"
  CUDA_VISIBLE_DEVICES=1 \
  CUDA_HOME=/usr/local/cuda-12.1 \
  TORCH_CUDA_ARCH_LIST=8.6 \
  MAX_JOBS=4 \
  "$PYTHON" -m pip install -e "$CUDA_OPS_ROOT" \
    --no-build-isolation --no-deps 2>&1 | tee "$LOG_ROOT/build_vis4d_cuda_ops.log"
fi
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -c \
  'import torch; import vis4d_cuda_ops; print(vis4d_cuda_ops.__file__)' \
  2>&1 | tee "$LOG_ROOT/vis4d_cuda_ops_import.log"

status "GPU gate passed; starting two-frame ScanNet smoke test on GPU 1"
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "$PYTHON" "$RUNNER" \
  --dataset scannet \
  --scene scene0568_00 \
  --output-root "$SMOKE_OUT" \
  --model-cache "$MODEL_CACHE" \
  --max-frames 2 \
  --no-resume 2>&1 | tee "$LOG_ROOT/smoke_scannet.log"
status "smoke test passed"

run_two_shards() {
  local dataset="$1"
  local output_root="$2"
  local dataset_log="$LOG_ROOT/$dataset"
  mkdir -p "$output_root" "$dataset_log"

  status "starting $dataset inference with one deterministic shard per GPU"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON" "$RUNNER" \
    --dataset "$dataset" \
    --output-root "$output_root" \
    --model-cache "$MODEL_CACHE" \
    --num-shards 2 \
    --shard-index 0 2>&1 | tee "$dataset_log/gpu0.log" &
  local pid0=$!

  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "$PYTHON" "$RUNNER" \
    --dataset "$dataset" \
    --output-root "$output_root" \
    --model-cache "$MODEL_CACHE" \
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
status "running the existing ScanNetV2-100 class-agnostic evaluator"
bash "$ROOT/scripts/eval_scannet_official100_real_score.sh" \
  mapdet3d_scannet100 "$SCANNET_OUT" 2>&1 | tee "$LOG_ROOT/eval_scannet100.log"

run_two_shards ca1m "$CA1M_OUT"
ca1m_count="$(find "$CA1M_OUT" -maxdepth 1 -type f -name '*_boxes.pkl' | wc -l)"
[[ "$ca1m_count" -eq 107 ]] || {
  status "ERROR expected 107 CA-1M predictions, found $ca1m_count"
  exit 1
}
status "running the existing CA-1M-107 class-agnostic evaluator"
/home/admin1/miniconda3/envs/boxfusion2/bin/python "$ROOT/tools/eval_ca1m.py" \
  --pred-root "$CA1M_OUT" 2>&1 | tee "$LOG_ROOT/eval_ca1m107.log"

status "COMPLETE: Map-Det3D ScanNetV2-100 and CA-1M-107 results are ready"
