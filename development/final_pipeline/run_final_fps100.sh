#!/usr/bin/env bash
set -euo pipefail

CODE=/data/ZhaoX/BoxFusion/development/final_runtime
MAIN=/data/ZhaoX/BoxFusion
PIPE=/data/ZhaoX/BoxFusion/development/final_pipeline
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
CONFIG="$CODE/config/scannet_recar3d_final_runtime.yaml"
SCENE_LIST="$MAIN/evaluation/data_util/meta_data/scannetv2_val.txt"
REPORT="$PIPE/reports/scannet_recar3d_final_fps100"
RECEIPTS="$REPORT/receipts"
LOGS="$REPORT/logs"
MPL="$REPORT/mplconfig"

mkdir -p "$RECEIPTS/control" "$RECEIPTS/full" "$LOGS" "$MPL"
exec 9>"$REPORT/run.lock"
flock -n 9 || { echo "Final FPS runner is active" >&2; exit 1; }
exec > >(tee -a "$REPORT/driver.log") 2>&1
mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 100 ]]

fingerprint() {
  sha256sum "$CONFIG" "$CODE/demo.py" "$PIPE/run_final_fps100.sh" \
    "$PIPE/summarize_final_fps.py" "$CODE/tools/benchmark_current_boxfusion_scene.py" \
    "$CODE/boxfusion/reliability_candidate_map.py" \
    "$CODE/boxfusion/native_reliability_reranker.py" \
    "$CODE/boxfusion/causal_reliability.py" \
    "$CODE/boxfusion/m1_anchor_online.py" \
    "$CODE/boxfusion/online_candidate_map.py" \
    "$CODE/boxfusion/online_candidate_runtime.py" | sha256sum | awk '{print $1}'
}
RUN_FINGERPRINT="$(fingerprint)"
if [[ -s "$REPORT/source.fingerprint" ]]; then
  [[ "$(tr -d '\n' < "$REPORT/source.fingerprint")" == "$RUN_FINGERPRINT" ]] || {
    echo "Final runtime fingerprint changed" >&2; exit 1;
  }
else
  printf '%s\n' "$RUN_FINGERPRINT" > "$REPORT/source.fingerprint"
fi

valid_arm() {
  local arm="$1" scene="$2" path="$RECEIPTS/$arm/$scene/runtime.json"
  [[ -s "$path" ]] || return 1
  "$PYTHON" - "$path" "$arm" "$scene" <<'PY' >/dev/null
import json, pathlib, sys
p, arm, scene=sys.argv[1:]; d=json.load(open(p))
assert d['arm']==arm and d['scene']==scene and d['gpu_count']==1
assert d['cuda_synchronized'] is True and d['scene_cache_prewarmed'] is True
assert d['final_output_serialization_included'] is True
assert d['model_initialization_excluded'] is True and d['visualization'] is False
if arm == 'full':
    diag=pathlib.Path(p).parent/'full_diagnostics'/f'{scene}.json'
    s=json.load(open(diag))['state']
    assert s['selected_calr_v1'] is True
    assert s['m1p']['use_children'] is False
    assert s['m2']['support_mode']=='max'
PY
}

run_arm() {
  local arm="$1"
  local scene="$2"
  local out="$RECEIPTS/$arm/$scene"
  if valid_arm "$arm" "$scene"; then echo "Reusing arm=$arm scene=$scene"; return; fi
  [[ ! -e "$out/runtime.json" ]] || { echo "Invalid receipt: $out" >&2; exit 1; }
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
    MPLCONFIGDIR="$MPL" \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
    PYTHONPATH="$CODE:$MAIN/third_party/WeDetect" \
    "$PYTHON" "$CODE/tools/benchmark_current_boxfusion_scene.py" \
      --arm "$arm" --scene "$scene" --config "$CONFIG" \
      --output-root "$out" --device-index 0 > "$LOGS/${scene}.${arm}.log" 2>&1
  valid_arm "$arm" "$scene"
}

echo "[$(date '+%F %T')] final matched FPS100 started fingerprint=$RUN_FINGERPRINT"
for index in "${!SCENES[@]}"; do
  [[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || { echo "Source changed" >&2; exit 1; }
  scene="${SCENES[$index]}"
  echo "[$(date '+%F %T')] [$((index+1))/100] $scene"
  if (( index % 2 == 0 )); then
    run_arm control "$scene"; run_arm full "$scene"
  else
    run_arm full "$scene"; run_arm control "$scene"
  fi
done
"$PYTHON" "$PIPE/summarize_final_fps.py" \
  --scene-list "$SCENE_LIST" --receipt-root "$RECEIPTS" --output "$REPORT"
echo "[$(date '+%F %T')] final matched FPS100 complete: $REPORT/REPORT.md"
