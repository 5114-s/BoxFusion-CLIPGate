#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/ZhaoX/BoxFusion
DEV="$MAIN/development/final_controls"
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
PREPARED_ROOT="$MAIN/data/3rscan_boxfusion_val157"
SCENE_LIST="$PREPARED_ROOT/scenes.txt"
BASE_RUN="$MAIN/reports/3rscan_val157_recar3d_vs_original_20260928"
BASE_CONFIG="$BASE_RUN/configs/recar3d_locked.yaml"
MODEL="$MAIN/models/cutr_rgbd.pth"
CLIP_MODEL="$MAIN/models/open_clip_pytorch_model.bin"
CLASS_TXT="$MAIN/data/panoptic_categories_nomerge.txt"
RUN_TAG="${RUN_TAG:-3rscan157_novis_20261006}"
SCENE_LIMIT="${SCENE_LIMIT:-157}"
RUN_ROOT="$DEV/experiments/$RUN_TAG"
CONFIG="$RUN_ROOT/config.yaml"
LOG_ROOT="$RUN_ROOT/logs"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
COMPONENT_ROOT="$RUN_ROOT/components"
SCENE_TIMEOUT_S=2400

for required in "$PYTHON" "$BASE_CONFIG" "$SCENE_LIST" "$MODEL" "$CLIP_MODEL" "$CLASS_TXT"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
[[ "$SCENE_LIMIT" =~ ^[0-9]+$ ]] && (( SCENE_LIMIT >= 1 && SCENE_LIMIT <= 157 )) || {
  echo "SCENE_LIMIT must be in [1,157]" >&2; exit 1;
}
mapfile -t ALL_SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#ALL_SCENES[@]}" -eq 157 ]] || { echo "Expected 3RScan-157" >&2; exit 1; }
SCENES=("${ALL_SCENES[@]:0:$SCENE_LIMIT}")
mkdir -p "$SCENE_LOG_ROOT" "$RUN_ROOT/mplconfig" "$RUN_ROOT/predictions/native" \
  "$RUN_ROOT/predictions/online_host" "$RUN_ROOT/diagnostics/online" \
  "$RUN_ROOT/evidence" "$COMPONENT_ROOT"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "Runner is active: $RUN_TAG" >&2; exit 1; }
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

if [[ ! -s "$CONFIG" ]]; then
  "$PYTHON" "$DEV/tools/prepare_crossdataset_novis_config.py" \
    --base "$BASE_CONFIG" --run-root "$RUN_ROOT" --output "$CONFIG"
fi

repair_empty() {
  local scene="$1" log="$SCENE_LOG_ROOT/$1.log" output="$RUN_ROOT/predictions/native/${1}_boxes.pkl"
  [[ -e "$output" ]] && return 0
  grep -Fq 'Online runtime receipt |' "$log" || return 1
  [[ -s "$COMPONENT_ROOT/${scene}.json" ]] || return 1
  "$PYTHON" - "$COMPONENT_ROOT/${scene}.json" "$output" <<'PY' >/dev/null
import json, pickle, sys
m=json.load(open(sys.argv[1], encoding='utf-8'))
assert all(int(v)==0 for v in m['counts'].values())
with open(sys.argv[2], 'wb') as h: pickle.dump([[]], h)
PY
}

validate_scene() {
  local scene="$1" log="$SCENE_LOG_ROOT/$1.log" component
  [[ -s "$RUN_ROOT/predictions/native/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$RUN_ROOT/predictions/online_host/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$RUN_ROOT/diagnostics/online/${scene}.json" ]] || return 1
  [[ -s "$COMPONENT_ROOT/${scene}.json" && -s "$RUN_ROOT/evidence/${scene}.npz" ]] || return 1
  [[ -s "$log" && "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|RuntimeError|CUDA out of memory' "$log" || return 1
  for component in native_native native_max native_factor_no_visibility \
    births_plr_v1 births_calr_v1; do
    [[ -s "$COMPONENT_ROOT/$component/${scene}_boxes.pkl" ]] || return 1
  done
}

echo "[$(date '+%F %T')] 3RScan-157 no-visibility run started ($SCENE_LIMIT scenes)"
for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"; log="$SCENE_LOG_ROOT/$scene.log"
  if validate_scene "$scene"; then
    echo "[$(date '+%F %T')] [$((index+1))/$SCENE_LIMIT] Reusing $scene"; continue
  fi
  [[ ! -e "$log" ]] || { echo "Invalid partial log exists: $log" >&2; exit 1; }
  echo "[$(date '+%F %T')] [$((index+1))/$SCENE_LIMIT] Running $scene"
  (
    cd "$DEV"
    CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
    MPLCONFIGDIR="$RUN_ROOT/mplconfig" \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
    PYTHONPATH="$DEV:$MAIN/third_party/WeDetect" \
    timeout "$SCENE_TIMEOUT_S" "$PYTHON" demo.py scannet --model-path "$MODEL" \
      --clip_path "$CLIP_MODEL" --class_txt "$CLASS_TXT" --config "$CONFIG" \
      --device cuda:0 --seq "$scene"
  ) > "$log" 2>&1
  repair_empty "$scene" "$log" || true
  validate_scene "$scene" || { echo "Scene validation failed: $scene" >&2; exit 1; }
  echo "[$(date '+%F %T')] [$((index+1))/$SCENE_LIMIT] Completed $scene"
done

if (( SCENE_LIMIT < 157 )); then
  echo "[$(date '+%F %T')] smoke run complete"; exit 0
fi

for spec in four_factor:native_max no_visibility:native_factor_no_visibility; do
  name="${spec%%:*}"; component="${spec#*:}"; root="$RUN_ROOT/factorial_$name"
  [[ ! -e "$root/manifest.json" ]] || { echo "Factorial exists: $root" >&2; exit 1; }
  "$PYTHON" "$MAIN/tools/materialize_3rscan_recar3d_factorial.py" \
    --scene-list "$SCENE_LIST" --components "$COMPONENT_ROOT" --output "$root" \
    --mvsr-component "$component"
done
"$PYTHON" "$MAIN/tools/eval_3rscan_boxfusion.py" \
  --prepared-root "$PREPARED_ROOT" \
  --prediction previous_four_factor "$BASE_RUN/predictions/factorial/full" \
  --prediction paired_four_factor "$RUN_ROOT/factorial_four_factor/full" \
  --prediction no_visibility "$RUN_ROOT/factorial_no_visibility/full" \
  --output "$RUN_ROOT/report"
echo "[$(date '+%F %T')] 3RScan-157 no-visibility comparison complete"
