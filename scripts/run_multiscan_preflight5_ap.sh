#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/ZhaoX/BoxFusion
DEV="$MAIN/development/reliability_v2"
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
RAW_ROOT=/extra/ZhaoX/MultiScan/scans
PREPARED_ROOT="$MAIN/data/multiscan_boxfusion_preflight5"
SCENE_LIST="$MAIN/data/multiscan_preflight5.txt"
RUN_ROOT="$MAIN/reports/multiscan_preflight5_20260930"
MODEL="$MAIN/models/cutr_rgbd.pth"
CLIP_MODEL="$MAIN/models/open_clip_pytorch_model.bin"
CLASS_TXT="$MAIN/data/panoptic_categories_nomerge.txt"
LOG_ROOT="$RUN_ROOT/logs"
SCENE_TIMEOUT_S=2400

mkdir -p "$RUN_ROOT" "$LOG_ROOT/original_boxfusion" "$LOG_ROOT/recar3d" "$RUN_ROOT/mplconfig"
exec 9>"$RUN_ROOT/run.lock"
flock -n 9 || { echo "driver already active" >&2; exit 1; }
exec > >(tee -a "$RUN_ROOT/driver.log") 2>&1

stamp() { date '+%F %T'; }
echo "[$(stamp)] MultiScan five-scene AP preflight started"

if [[ ! -s "$PREPARED_ROOT/manifest.json" ]]; then
  "$PYTHON" "$MAIN/tools/prepare_multiscan_preflight.py" \
    --raw-root "$RAW_ROOT" --output-root "$PREPARED_ROOT" \
    --scene-list "$SCENE_LIST" --stride 60 --resume
fi
mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$PREPARED_ROOT/scenes.txt")
[[ "${#SCENES[@]}" -eq 5 ]] || { echo "expected 5 prepared scenes, got ${#SCENES[@]}" >&2; exit 1; }

"$PYTHON" "$MAIN/tools/build_multiscan_recar3d_configs.py" \
  --prepared-root "$PREPARED_ROOT" --run-root "$RUN_ROOT"
RECAR_CONFIG="$RUN_ROOT/configs/recar3d_locked.yaml"
ORIGINAL_CONFIG="$RUN_ROOT/configs/original_boxfusion.yaml"

run_scene() {
  local arm="$1" tree="$2" config="$3" scene="$4" log="$5" gpu="$6"
  (
    cd "$tree"
    CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
    MPLCONFIGDIR="$RUN_ROOT/mplconfig" \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
    PYTHONPATH="$tree:$MAIN/third_party/WeDetect" \
    timeout "$SCENE_TIMEOUT_S" "$PYTHON" demo.py scannet --model-path "$MODEL" \
      --clip_path "$CLIP_MODEL" --class_txt "$CLASS_TXT" --config "$config" \
      --device cuda:0 --seq "$scene"
  ) >"$log" 2>&1
}

repair_empty() {
  local output="$1" log="$2"
  [[ -e "$output" ]] && return 0
  grep -Fq 'Online runtime receipt |' "$log" || return 1
  "$PYTHON" -c "import pickle; pickle.dump([[]], open('$output','wb'), protocol=pickle.HIGHEST_PROTOCOL)"
}

for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"
  original_out="$RUN_ROOT/predictions/original_boxfusion/${scene}_boxes.pkl"
  recar_out="$RUN_ROOT/predictions/recar3d_native/${scene}_boxes.pkl"
  original_log="$LOG_ROOT/original_boxfusion/${scene}.log"
  recar_log="$LOG_ROOT/recar3d/${scene}.log"
  echo "[$(stamp)] [$((index+1))/5] running $scene (ReCaR-3D GPU0, original GPU1)"
  if [[ ! -s "$original_out" ]]; then
    run_scene original "$MAIN" "$ORIGINAL_CONFIG" "$scene" "$original_log" 0 || { echo "original failed: $scene" >&2; exit 1; }
  fi
  if [[ ! -s "$recar_out" || ! -s "$RUN_ROOT/components/${scene}.json" ]]; then
    run_scene recar3d "$DEV" "$RECAR_CONFIG" "$scene" "$recar_log" 0,1 || { echo "ReCaR-3D failed: $scene" >&2; exit 1; }
  fi
  repair_empty "$original_out" "$original_log"
  repair_empty "$recar_out" "$recar_log"
  [[ -s "$original_out" && -s "$recar_out" && -s "$RUN_ROOT/components/${scene}.json" ]] \
    || { echo "missing completed artifacts: $scene" >&2; exit 1; }
  echo "[$(stamp)] [$((index+1))/5] completed $scene"
done

FACTORIAL_ROOT="$RUN_ROOT/predictions/factorial"
if [[ ! -s "$FACTORIAL_ROOT/manifest.json" ]]; then
  "$PYTHON" "$MAIN/tools/materialize_3rscan_recar3d_factorial.py" \
    --scene-list "$PREPARED_ROOT/scenes.txt" \
    --components "$RUN_ROOT/components" --output "$FACTORIAL_ROOT"
fi
"$PYTHON" "$MAIN/tools/eval_multiscan_boxfusion.py" \
  --prepared-root "$PREPARED_ROOT" \
  --prediction original_boxfusion "$RUN_ROOT/predictions/original_boxfusion" \
  --prediction recar3d_base "$FACTORIAL_ROOT/base" \
  --prediction recar3d_mvsr "$FACTORIAL_ROOT/mvsr" \
  --prediction recar3d_full "$FACTORIAL_ROOT/full" \
  --output "$RUN_ROOT/report"

sha256sum \
  "$MAIN/tools/prepare_multiscan_preflight.py" \
  "$MAIN/tools/build_multiscan_recar3d_configs.py" \
  "$MAIN/tools/eval_multiscan_boxfusion.py" \
  "$MAIN/tools/materialize_3rscan_recar3d_factorial.py" \
  "$ORIGINAL_CONFIG" "$RECAR_CONFIG" > "$RUN_ROOT/source_fingerprints.sha256"
echo "[$(stamp)] driver complete: $RUN_ROOT/report/REPORT.md"
