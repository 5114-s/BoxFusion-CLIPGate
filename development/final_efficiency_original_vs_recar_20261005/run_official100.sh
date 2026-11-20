#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
DEV="$ROOT/development/final_efficiency_original_vs_recar_20261005"
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
ORIGINAL="$ROOT/reports/paper_matched_runtime_full100_20260916/official_source_06ea629"
RECAR="$DEV/recar_source"
CONFIG="$RECAR/config/scannet_recar3d_final_runtime.yaml"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
REPORT="$ROOT/reports/original_boxfusion_vs_final_recar_efficiency100_20261005"
RECEIPTS="$REPORT/receipts"
LOGS="$REPORT/logs"
MPL="$REPORT/mplconfig"

mkdir -p "$RECEIPTS/original" "$RECEIPTS/recar" "$LOGS" "$MPL"
exec 9>"$REPORT/run.lock"
flock -n 9 || { echo "efficiency runner is already active" >&2; exit 1; }
exec > >(tee -a "$REPORT/driver.log") 2>&1

mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 100 ]]

fingerprint() {
  find "$DEV/tools" "$RECAR" "$ORIGINAL" -type f \
    ! -path '*/__pycache__/*' ! -path '*/.pytest_cache/*' \
    ! -name '*.pyc' -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    | sha256sum \
    | awk '{print $1}'
}

RUN_FINGERPRINT="$(fingerprint)"
printf '%s\n' "$RUN_FINGERPRINT" > "$REPORT/source.fingerprint"
sha256sum "$SCENE_LIST" > "$REPORT/scene_list.sha256"
{
  echo "created=$(date --iso-8601=seconds)"
  echo "gpu_visible=0"
  echo "protocol=original_raw_frame_loop_no_explicit_cache_prewarm"
  echo "original_source=$ORIGINAL"
  echo "recar_source=$RECAR"
  echo "recar_git_archive=ad52642cc6d628ba0f52208181230c33a152c77e"
  echo "recar_route=PLR-v1+CALR-v1+MVSR-v2-Max"
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
} > "$REPORT/environment.txt"

valid_arm() {
  local arm="$1"
  local scene="$2"
  local path="$RECEIPTS/$arm/$scene/runtime.json"
  [[ -s "$path" ]] || return 1
  "$PYTHON" - "$path" "$arm" "$scene" <<'PY' >/dev/null
import json, sys
p, arm, scene = sys.argv[1:]
d = json.load(open(p, encoding="utf-8"))
assert d["arm"] == arm and d["scene"] == scene
assert d["gpu_count"] == 1 and d["gap"] == 25
assert d["score_threshold"] == 0.5
assert d["cuda_synchronized"] is True
assert d["prepare_all_raw_frames"] is True
assert d["scene_cache_prewarmed"] is False
assert d["model_initialization_excluded"] is True
assert d["visualization"] is False and d["proposal_replay"] is False
assert d["final_output_serialization_included"] is True
assert d["raw_frames"] > 0 and d["cost_seconds"] > 0 and d["fps"] > 0
PY
}

run_arm() {
  local arm="$1"
  local scene="$2"
  local source="$ORIGINAL"
  local extra=()
  if [[ "$arm" == "recar" ]]; then
    source="$RECAR"
    extra=(--config "$CONFIG")
  fi
  local out="$RECEIPTS/$arm/$scene"
  if valid_arm "$arm" "$scene"; then
    echo "Reusing arm=$arm scene=$scene"
    return
  fi
  [[ ! -e "$out/runtime.json" ]] || {
    echo "Invalid partial receipt: $out/runtime.json" >&2
    exit 1
  }
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
    MPLCONFIGDIR="$MPL" \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
    PYTHONPATH="$source:$ROOT/third_party/WeDetect" \
    "$PYTHON" "$DEV/tools/benchmark_scene.py" \
      --arm "$arm" --scene "$scene" --source-root "$source" \
      --repository-root "$ROOT" --output-root "$out" \
      --device-index 0 "${extra[@]}" \
      > "$LOGS/${scene}.${arm}.log" 2>&1
  valid_arm "$arm" "$scene"
}

echo "[$(date '+%F %T')] original-vs-final-ReCaR official100 started fingerprint=$RUN_FINGERPRINT"
for index in "${!SCENES[@]}"; do
  [[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || {
    echo "Frozen source changed during run" >&2
    exit 1
  }
  scene="${SCENES[$index]}"
  echo "[$(date '+%F %T')] [$((index + 1))/100] $scene"
  if (( index % 2 == 0 )); then
    run_arm original "$scene"
    run_arm recar "$scene"
  else
    run_arm recar "$scene"
    run_arm original "$scene"
  fi
  original_fps="$("$PYTHON" -c "import json; print(json.load(open('$RECEIPTS/original/$scene/runtime.json'))['fps'])")"
  recar_fps="$("$PYTHON" -c "import json; print(json.load(open('$RECEIPTS/recar/$scene/runtime.json'))['fps'])")"
  echo "[$(date '+%F %T')] Completed $scene original_fps=$original_fps recar_fps=$recar_fps"
done

"$PYTHON" "$DEV/tools/summarize.py" \
  --scene-list "$SCENE_LIST" --receipt-root "$RECEIPTS" --output "$REPORT"
echo "[$(date '+%F %T')] original-vs-final-ReCaR official100 complete"
