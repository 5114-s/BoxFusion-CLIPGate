#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
PYTHON=/home/admin1/miniconda3/envs/boxfusion-online/bin/python
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
CONFIG="$ROOT/config/scannet_t05_boxer_strict_causal_nochild_single_gpu_fps.yaml"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
OFFICIAL_SOURCE="$ROOT/reports/paper_matched_runtime_full100_20260916/official_source_06ea629"
REPORT="$ROOT/reports/strict_causal_nochild_single_gpu_matched_fps_20260923"
OFFICIAL="$REPORT/official"
LOGS="$REPORT/method_logs"
MPL="$REPORT/mplconfig"
MODEL="$ROOT/models/cutr_rgbd.pth"
CLIP="$ROOT/models/open_clip_pytorch_model.bin"
CLASSES="$ROOT/data/panoptic_categories_nomerge.txt"
METHOD_NATIVE="$ROOT/results/scannet_strict_causal_nochild_single_gpu_fps_native"
METHOD_FULL="$ROOT/results/scannet_strict_causal_nochild_single_gpu_fps"
METHOD_DIAG="$ROOT/diagnostics/scannet_strict_causal_nochild_single_gpu_fps"
METHOD_PROVIDER_DIAG="$ROOT/diagnostics/scannet_strict_causal_nochild_single_gpu_fps_provider"
METHOD_NATIVE_DIAG="$ROOT/diagnostics/scannet_strict_causal_nochild_single_gpu_fps_native"
METHOD_PVQ_DIAG="$ROOT/diagnostics/scannet_strict_causal_nochild_single_gpu_fps_pvq"

mkdir -p "$OFFICIAL" "$LOGS" "$MPL" "$METHOD_NATIVE" "$METHOD_FULL" \
  "$METHOD_DIAG" "$METHOD_PROVIDER_DIAG" "$METHOD_NATIVE_DIAG" \
  "$METHOD_PVQ_DIAG"
exec 9>"$REPORT/run.lock"
flock -n 9 || { echo "matched FPS runner is already active" >&2; exit 1; }
exec > >(tee -a "$REPORT/driver.log") 2>&1

mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 100 ]]

run_official() {
  local scene="$1" out="$OFFICIAL/$scene"
  [[ -s "$out/runtime.json" ]] && return
  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=8 MPLCONFIGDIR="$MPL" \
    "$PYTHON" "$ROOT/tools/benchmark_official_boxfusion_scene.py" \
      --source-root "$OFFICIAL_SOURCE" --repository-root "$ROOT" \
      --scene "$scene" --output-root "$out" --device-index 0 \
      > "$REPORT/${scene}.official.log" 2>&1
}

run_method() {
  local scene="$1" log="$LOGS/$scene.log"
  if [[ -s "$log" ]] && [[ "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]]; then
    return
  fi
  [[ ! -e "$log" ]] || { echo "invalid partial method log: $log" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
    MPLCONFIGDIR="$MPL" \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
    PYTHONPATH="$ROOT:$ROOT/third_party/WeDetect" \
    "$PYTHON" "$ROOT/demo.py" scannet \
      --model-path "$MODEL" --clip_path "$CLIP" --class_txt "$CLASSES" \
      --config "$CONFIG" --device cuda:0 --seq "$scene" > "$log" 2>&1
  [[ "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]]
  grep -Fq 'Online candidate map summary | causal=True scene_end_inference=False' "$log"
  ! grep -Eq 'Traceback|RuntimeError|CUDA out of memory' "$log"
}

echo "[$(date '+%F %T')] single-GPU matched FPS official100 started"
for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"
  echo "[$(date '+%F %T')] [$((index + 1))/100] $scene"
  if (( index % 2 == 0 )); then
    run_official "$scene"
    run_method "$scene"
  else
    run_method "$scene"
    run_official "$scene"
  fi
done

"$PYTHON" "$ROOT/tools/summarize_strict_online_matched_fps.py" \
  --scene-list "$SCENE_LIST" --official-root "$OFFICIAL" \
  --method-logs "$LOGS" --output "$REPORT"
echo "[$(date '+%F %T')] single-GPU matched FPS official100 complete"
