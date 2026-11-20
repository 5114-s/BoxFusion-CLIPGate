#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/ZhaoX/BoxFusion
DEV="$MAIN/development/final_controls"
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
PREPARED_ROOT="$MAIN/data/multiscan_boxfusion_eval50"
SCENE_LIST="$PREPARED_ROOT/scenes.txt"
BASE_RUN="$MAIN/reports/multiscan_eval50_calrv1_20261001"
BASE_CONFIG="$BASE_RUN/configs/recar3d_locked.yaml"
MODEL="$MAIN/models/cutr_rgbd.pth"
CLIP_MODEL="$MAIN/models/open_clip_pytorch_model.bin"
CLASS_TXT="$MAIN/data/panoptic_categories_nomerge.txt"
RUN_TAG="${RUN_TAG:-multiscan50_novis_20261006}"
SCENE_LIMIT="${SCENE_LIMIT:-50}"
RUN_ROOT="$DEV/experiments/$RUN_TAG"
CONFIG="$RUN_ROOT/config.yaml"
LOG_ROOT="$RUN_ROOT/logs"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
COMPONENT_ROOT="$RUN_ROOT/components"
SCENE_TIMEOUT_S=14400

for required in "$PYTHON" "$BASE_CONFIG" "$SCENE_LIST" "$MODEL" "$CLIP_MODEL" "$CLASS_TXT"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
[[ "$SCENE_LIMIT" =~ ^[0-9]+$ ]] && (( SCENE_LIMIT >= 1 && SCENE_LIMIT <= 50 )) || {
  echo "SCENE_LIMIT must be in [1,50]" >&2; exit 1;
}
mapfile -t ALL_SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#ALL_SCENES[@]}" -eq 50 ]] || { echo "Expected MultiScan-50" >&2; exit 1; }
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

archive_partial_scene() {
  local scene="$1" stamp archive path
  stamp="$(date '+%Y%m%d_%H%M%S')"
  archive="$RUN_ROOT/interrupted/$scene/$stamp"
  mkdir -p "$archive"
  for path in \
    "$SCENE_LOG_ROOT/$scene.log" \
    "$RUN_ROOT/predictions/native/${scene}_boxes.pkl" \
    "$RUN_ROOT/predictions/online_host/${scene}_boxes.pkl" \
    "$RUN_ROOT/diagnostics/online/${scene}.json" \
    "$RUN_ROOT/evidence/${scene}.npz" \
    "$COMPONENT_ROOT/${scene}.json" \
    "$RUN_ROOT/diagnostics/native/${scene}_boxer_lifting.jsonl" \
    "$RUN_ROOT/diagnostics/pvq/${scene}_pvq_ar.jsonl" \
    "$RUN_ROOT/diagnostics/pvq/${scene}_pvq_ar_summary.json" \
    "$RUN_ROOT/diagnostics/pvq/${scene}_pvq_kfmap.jsonl" \
    "$RUN_ROOT/diagnostics/pvq/${scene}_pvq_nms.jsonl" \
    "$RUN_ROOT/diagnostics/pvq/${scene}_pvq_nms_ar.jsonl"; do
    if [[ -e "$path" ]]; then
      mkdir -p "$archive/$(dirname "${path#$RUN_ROOT/}")"
      mv "$path" "$archive/${path#$RUN_ROOT/}"
    fi
  done
  while IFS= read -r -d '' path; do
    mkdir -p "$archive/$(dirname "${path#$RUN_ROOT/}")"
    mv "$path" "$archive/${path#$RUN_ROOT/}"
  done < <(find "$COMPONENT_ROOT" -mindepth 2 -maxdepth 2 -type f \
    -name "${scene}_boxes.pkl" -print0)
  echo "Archived interrupted artifacts for $scene to $archive"
}

echo "[$(date '+%F %T')] MultiScan-50 no-visibility run started ($SCENE_LIMIT scenes)"
for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"; log="$SCENE_LOG_ROOT/$scene.log"
  if validate_scene "$scene"; then
    echo "[$(date '+%F %T')] [$((index+1))/$SCENE_LIMIT] Reusing $scene"; continue
  fi
  if [[ -e "$log" ]]; then
    archive_partial_scene "$scene"
  fi
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

if (( SCENE_LIMIT < 50 )); then
  echo "[$(date '+%F %T')] smoke run complete"; exit 0
fi

for spec in four_factor:native_max no_visibility:native_factor_no_visibility; do
  name="${spec%%:*}"; component="${spec#*:}"; root="$RUN_ROOT/factorial_$name"
  if [[ ! -s "$root/manifest.json" ]]; then
    "$PYTHON" "$MAIN/tools/materialize_3rscan_recar3d_factorial.py" \
      --scene-list "$SCENE_LIST" --components "$COMPONENT_ROOT" --output "$root" \
      --mvsr-component "$component"
  else
    echo "Reusing factorial manifest: $root/manifest.json"
  fi
done
"$PYTHON" "$MAIN/tools/eval_multiscan_boxfusion.py" \
  --prepared-root "$PREPARED_ROOT" \
  --prediction previous_four_factor "$BASE_RUN/predictions/factorial/full" \
  --prediction paired_four_factor "$RUN_ROOT/factorial_four_factor/full" \
  --prediction base "$RUN_ROOT/factorial_no_visibility/base" \
  --prediction plr "$RUN_ROOT/factorial_no_visibility/plr" \
  --prediction calr "$RUN_ROOT/factorial_no_visibility/calr" \
  --prediction mvsr "$RUN_ROOT/factorial_no_visibility/mvsr" \
  --prediction plr_calr "$RUN_ROOT/factorial_no_visibility/plr_calr" \
  --prediction plr_mvsr "$RUN_ROOT/factorial_no_visibility/plr_mvsr" \
  --prediction calr_mvsr "$RUN_ROOT/factorial_no_visibility/calr_mvsr" \
  --prediction full_no_visibility "$RUN_ROOT/factorial_no_visibility/full" \
  --output "$RUN_ROOT/report"
"$PYTHON" "$MAIN/tools/eval_multiscan_calr_budget_bootstrap.py" \
  --prepared-root "$PREPARED_ROOT" \
  --components "$COMPONENT_ROOT" \
  --mvsr-component native_factor_no_visibility \
  --budgets 50 100 200 500 \
  --bootstraps 2000 --seed 20261006 \
  --output "$RUN_ROOT/budget_report"
echo "[$(date '+%F %T')] MultiScan-50 no-visibility comparison complete"
