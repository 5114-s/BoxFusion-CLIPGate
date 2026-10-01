#!/usr/bin/env bash
set -euo pipefail

DEV=/data/ZhaoX/BoxFusion/development/plr_v2
MAIN=/data/ZhaoX/BoxFusion
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
CONFIG="$DEV/config/scannet_t05_boxer_plr_v2_official100.yaml"
SCENE_LIST="$MAIN/evaluation/data_util/meta_data/scannetv2_val.txt"
MODEL="$MAIN/models/cutr_rgbd.pth"
CLIP_MODEL="$MAIN/models/open_clip_pytorch_model.bin"
CLASS_TXT="$MAIN/data/panoptic_categories_nomerge.txt"
NATIVE_ROOT="$DEV/results/scannet_plr_v2_official100_native"
ONLINE_ROOT="$DEV/results/scannet_plr_v2_official100"
DIAGNOSTICS_ROOT="$DEV/diagnostics/scannet_plr_v2_official100"
COMPONENT_ROOT="$DEV/results/scannet_plr_v2_components"
ARM_ROOT="$DEV/results/scannet_plr_v2_arms"
REPORT_ROOT="$DEV/reports/scannet_plr_v2_official100_20260926"
LOG_ROOT="$DEV/logs/scannet_plr_v2_official100"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
MPL_ROOT="$LOG_ROOT/mplconfig"
FINGERPRINT_FILE="$LOG_ROOT/source.fingerprint"

for required in "$PYTHON" "$CONFIG" "$SCENE_LIST" "$MODEL" "$CLIP_MODEL" "$CLASS_TXT"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
[[ "$(sha256sum "$SCENE_LIST" | awk '{print $1}')" == \
  4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5 ]] || {
  echo "Official100 scene-list hash mismatch" >&2; exit 1;
}
mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 100 ]] || { echo "Expected 100 scenes" >&2; exit 1; }
mkdir -p "$NATIVE_ROOT" "$ONLINE_ROOT" "$DIAGNOSTICS_ROOT" "$COMPONENT_ROOT" \
  "$SCENE_LOG_ROOT" "$MPL_ROOT" "$REPORT_ROOT"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "PLR-v2 runner is active" >&2; exit 1; }
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

fingerprint() {
  sha256sum "$CONFIG" "$DEV/demo.py" "$DEV/PLR_V2_PROTOCOL.md" \
    "$DEV/FROZEN_PREFIX.json" \
    "$DEV/scripts/run_plr_v2_official100.sh" \
    "$DEV/tools/materialize_plr_v2_arms.py" \
    "$DEV/tools/audit_plr_v2_screening.py" \
    "$DEV/boxfusion/plr_v2.py" \
    "$DEV/boxfusion/reliability_candidate_map.py" \
    "$DEV/boxfusion/native_reliability_reranker.py" \
    "$DEV/boxfusion/causal_reliability.py" \
    "$DEV/boxfusion/m1_anchor_reliability_online.py" \
    "$DEV/boxfusion/m1_anchor_online.py" \
    "$DEV/boxfusion/online_candidate_map.py" \
    "$DEV/boxfusion/online_candidate_runtime.py" \
    "$DEV/boxfusion/box_manager.py" "$DEV/boxfusion/instances.py" \
    "$MODEL" "$CLIP_MODEL" "$CLASS_TXT" | sha256sum | awk '{print $1}'
}
RUN_FINGERPRINT="$(fingerprint)"
if [[ -s "$FINGERPRINT_FILE" ]]; then
  [[ "$(tr -d '\n' < "$FINGERPRINT_FILE")" == "$RUN_FINGERPRINT" ]] || {
    echo "Source/config fingerprint differs from existing run" >&2; exit 1;
  }
else
  printf '%s\n' "$RUN_FINGERPRINT" > "$FINGERPRINT_FILE"
fi

COMPONENTS=(native_native native_first native_mean native_max native_ema \
  native_diverse_max native_reliability births_plr_v1 births_plr_assoc \
  births_plr_reliability births_plr_score births_calr_v1 births_calr_v2)

validate_scene() {
  local scene="$1" log="$SCENE_LOG_ROOT/$1.log"
  [[ -s "$NATIVE_ROOT/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$ONLINE_ROOT/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$DIAGNOSTICS_ROOT/${scene}.json" ]] || return 1
  [[ -s "$COMPONENT_ROOT/${scene}.json" ]] || return 1
  [[ -s "$log" ]] || return 1
  [[ "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|RuntimeError|CUDA out of memory' "$log" || return 1
  local component
  for component in "${COMPONENTS[@]}"; do
    [[ -s "$COMPONENT_ROOT/$component/${scene}_boxes.pkl" ]] || return 1
  done
  "$PYTHON" - "$DIAGNOSTICS_ROOT/${scene}.json" "$COMPONENT_ROOT/${scene}.json" <<'PY' >/dev/null
import json, sys
diagnostic=json.load(open(sys.argv[1],encoding='utf-8'))
components=json.load(open(sys.argv[2],encoding='utf-8'))
state=diagnostic['state']
assert diagnostic['strictly_causal'] is True
assert diagnostic['online_incremental'] is True
assert diagnostic['scene_end_inference'] is False
assert state['cross_branch_dedup'] is False
assert state['m1p']['use_children'] is False
assert state['m1a_control'] is not None
assert state['m2']['support_mode'] == 'max'
assert tuple(sorted(state['plr_controls'])) == ('assoc','reliability','score')
assert all(row['use_children'] is False for row in state['plr_controls'].values())
assert components['single_online_pass'] is True
assert components['future_frames_used'] is False
assert len(components['counts']) == 13
PY
}

echo "[$(date '+%F %T')] PLR-v2 official100 started fingerprint=$RUN_FINGERPRINT"
for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"; log="$SCENE_LOG_ROOT/$scene.log"
  [[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || {
    echo "Source/config changed before $scene" >&2; exit 1;
  }
  if validate_scene "$scene"; then
    echo "[$(date '+%F %T')] [$((index+1))/100] Reusing $scene"; continue
  fi
  for stale in "$NATIVE_ROOT/${scene}_boxes.pkl" "$ONLINE_ROOT/${scene}_boxes.pkl" \
    "$DIAGNOSTICS_ROOT/${scene}.json" "$COMPONENT_ROOT/${scene}.json" "$log"; do
    [[ ! -e "$stale" ]] || { echo "Invalid partial artifact: $stale" >&2; exit 1; }
  done
  for component in "${COMPONENTS[@]}"; do
    [[ ! -e "$COMPONENT_ROOT/$component/${scene}_boxes.pkl" ]] || {
      echo "Invalid partial component: $component/$scene" >&2; exit 1;
    }
  done
  echo "[$(date '+%F %T')] [$((index+1))/100] Running $scene"
  (
    cd "$DEV"
    CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
    MPLCONFIGDIR="$MPL_ROOT" \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
    PYTHONPATH="$DEV:$MAIN/third_party/WeDetect" \
    "$PYTHON" demo.py scannet --model-path "$MODEL" --clip_path "$CLIP_MODEL" \
      --class_txt "$CLASS_TXT" --config "$CONFIG" --device cuda:0 --seq "$scene"
  ) > "$log" 2>&1
  validate_scene "$scene" || { echo "Scene validation failed: $scene" >&2; exit 1; }
  echo "[$(date '+%F %T')] [$((index+1))/100] Completed $scene"
done

[[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || { echo "Source changed" >&2; exit 1; }
[[ ! -e "$ARM_ROOT/manifest.json" ]] || { echo "Arm manifest exists" >&2; exit 1; }
"$PYTHON" "$DEV/tools/materialize_plr_v2_arms.py" \
  --scene-list "$SCENE_LIST" --components "$COMPONENT_ROOT" --output "$ARM_ROOT"
"$PYTHON" "$DEV/tools/audit_plr_v2_screening.py" \
  --scene-list "$SCENE_LIST" --arms "$ARM_ROOT" \
  --diagnostics "$DIAGNOSTICS_ROOT" --output "$REPORT_ROOT"

OFFICIAL_ARMS=(prefix prefix_plr_current prefix_plr_assoc \
  prefix_plr_reliability prefix_plr_score)
for arm in "${OFFICIAL_ARMS[@]}"; do
  bash "$MAIN/scripts/eval_scannet_official100_real_score.sh" \
    "plr_v2_paired_${arm}" "$ARM_ROOT/$arm"
done
echo "[$(date '+%F %T')] PLR-v2 official100 complete: $REPORT_ROOT/REPORT.md"
