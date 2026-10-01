#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
PYTHON=/home/admin1/miniconda3/envs/boxfusion-online/bin/python
PREPARED_ROOT="${1:?Usage: $0 PREPARED_ROOT RUN_ROOT}"
RUN_ROOT="${2:?Usage: $0 PREPARED_ROOT RUN_ROOT}"
GPU="${THREERSCAN_GPU:-1}"
MODEL="$ROOT/models/cutr_rgbd.pth"
CLIP_MODEL="$ROOT/models/open_clip_pytorch_model.bin"
CLASS_TXT="$ROOT/data/panoptic_categories_nomerge.txt"

for path in "$PREPARED_ROOT/manifest.json" "$PREPARED_ROOT/scenes.txt" "$MODEL" "$CLIP_MODEL" "$CLASS_TXT"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done

mkdir -p "$RUN_ROOT"
"$PYTHON" "$ROOT/tools/build_3rscan_boxfusion_configs.py" \
  --prepared-root "$PREPARED_ROOT" --run-root "$RUN_ROOT"
ORIGINAL_CONFIG="$RUN_ROOT/configs/original_boxfusion.yaml"
LATEST_CONFIG="$RUN_ROOT/configs/latest_full.yaml"

run_scene() {
  local arm="$1" config="$2" scene="$3" output="$4" log="$5"
  if [[ -s "$output/${scene}_boxes.pkl" ]]; then
    echo "Reusing $arm $scene"
    return
  fi
  echo "Running $arm $scene"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
    MPLCONFIGDIR="$RUN_ROOT/mplconfig" \
    LD_LIBRARY_PATH="/home/admin1/miniconda3/envs/boxfusion-online/lib:/usr/local/cuda-12.1/lib64" \
    PYTHONPATH="$ROOT:$ROOT/third_party/WeDetect" \
    "$PYTHON" demo.py scannet --model-path "$MODEL" --clip_path "$CLIP_MODEL" \
      --class_txt "$CLASS_TXT" --config "$config" --device cuda:0 --seq "$scene"
  ) >"$log" 2>&1
  [[ -s "$output/${scene}_boxes.pkl" ]] || { echo "Missing output after $arm $scene; see $log" >&2; exit 1; }
}

mkdir -p "$RUN_ROOT/mplconfig"
while IFS= read -r scene || [[ -n "$scene" ]]; do
  [[ -n "$scene" ]] || continue
  run_scene original_boxfusion "$ORIGINAL_CONFIG" "$scene" \
    "$RUN_ROOT/predictions/original_boxfusion" "$RUN_ROOT/logs/original_boxfusion/${scene}.log"
  run_scene latest_full "$LATEST_CONFIG" "$scene" \
    "$RUN_ROOT/predictions/latest_full" "$RUN_ROOT/logs/latest_full/${scene}.log"
  [[ -s "$RUN_ROOT/predictions/latest_native_host/${scene}_boxes.pkl" ]] || {
    echo "Latest run did not emit native host output for $scene" >&2; exit 1;
  }
done < "$PREPARED_ROOT/scenes.txt"

"$PYTHON" "$ROOT/tools/eval_3rscan_boxfusion.py" \
  --prepared-root "$PREPARED_ROOT" \
  --prediction original_boxfusion "$RUN_ROOT/predictions/original_boxfusion" \
  --prediction latest_native_host "$RUN_ROOT/predictions/latest_native_host" \
  --prediction latest_full_nochild "$RUN_ROOT/predictions/latest_full" \
  --output "$RUN_ROOT/report"
