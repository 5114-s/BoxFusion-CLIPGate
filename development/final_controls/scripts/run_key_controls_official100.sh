#!/usr/bin/env bash
set -euo pipefail

DEV=/data/ZhaoX/BoxFusion/development/final_controls
MAIN=/data/ZhaoX/BoxFusion
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
CONFIG="$DEV/config/scannet_recar3d_key_controls_official100.yaml"
SCENE_LIST="$MAIN/evaluation/data_util/meta_data/scannetv2_val.txt"
MODEL="$MAIN/models/cutr_rgbd.pth"
CLIP_MODEL="$MAIN/models/open_clip_pytorch_model.bin"
CLASS_TXT="$MAIN/data/panoptic_categories_nomerge.txt"
NATIVE_ROOT="$DEV/results/scannet_recar3d_key_controls_native"
ONLINE_ROOT="$DEV/results/scannet_recar3d_key_controls"
DIAGNOSTICS_ROOT="$DEV/diagnostics/scannet_recar3d_key_controls"
COMPONENT_ROOT="$DEV/results/scannet_recar3d_key_control_components"
EVIDENCE_ROOT="$DEV/evidence/scannet_recar3d_key_controls"
ARM_ROOT="$DEV/results/scannet_recar3d_key_control_arms"
CALR_ROOT="$DEV/results/scannet_recar3d_calr_controls"
REPORT_ROOT="$DEV/reports/scannet_recar3d_key_controls_official100_20260926"
LOG_ROOT="$DEV/logs/scannet_recar3d_key_controls_official100"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
MPL_ROOT="$LOG_ROOT/mplconfig"
FINGERPRINT_FILE="$LOG_ROOT/source.fingerprint"
EVAL_LOG_ROOT="$MAIN/logs/scannet_official100_real_score"

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
  "$EVIDENCE_ROOT" "$SCENE_LOG_ROOT" "$MPL_ROOT" "$REPORT_ROOT"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "Final key-control runner is active" >&2; exit 1; }
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

fingerprint() {
  sha256sum "$CONFIG" "$DEV/demo.py" \
    "$DEV/scripts/run_key_controls_official100.sh" \
    "$DEV/tools/materialize_key_controls.py" \
    "$DEV/tools/materialize_calr_key_controls.py" \
    "$DEV/tools/summarize_key_controls.py" \
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
  native_diverse_max native_reliability native_raw_iou_max \
  births_plr_v1 births_plr_direct births_calr_v1 births_calr_v2)

validate_scene() {
  local scene="$1" log="$SCENE_LOG_ROOT/$1.log"
  [[ -s "$NATIVE_ROOT/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$ONLINE_ROOT/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$DIAGNOSTICS_ROOT/${scene}.json" ]] || return 1
  [[ -s "$COMPONENT_ROOT/${scene}.json" ]] || return 1
  [[ -s "$EVIDENCE_ROOT/${scene}.npz" ]] || return 1
  [[ -s "$log" ]] || return 1
  [[ "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|RuntimeError|CUDA out of memory' "$log" || return 1
  local component
  for component in "${COMPONENTS[@]}"; do
    [[ -s "$COMPONENT_ROOT/$component/${scene}_boxes.pkl" ]] || return 1
  done
  "$PYTHON" - "$DIAGNOSTICS_ROOT/${scene}.json" "$COMPONENT_ROOT/${scene}.json" <<'PY' >/dev/null
import json, pickle, pathlib, sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
c=json.load(open(sys.argv[2],encoding='utf-8'))
s=d['state']
assert d['strictly_causal'] is True and d['online_incremental'] is True
assert d['scene_end_inference'] is False and s['strictly_causal'] is True
assert s['uses_future_frames'] is False and s['m1p']['use_children'] is False
assert s['m1a_control'] is not None and s['m2']['support_mode'] == 'max'
assert s['direct_plr_control']['temporal_confirmation_disabled'] is True
assert s['raw_m2_control'] is not None and s['raw_m2_control']['frames_seen'] == s['frames_seen']
assert c['single_online_pass'] is True and c['future_frames_used'] is False
root=pathlib.Path(sys.argv[2]).parent
scene=pathlib.Path(sys.argv[2]).stem
def count(name):
    with (root/name/(scene+'_boxes.pkl')).open('rb') as h:
        return len(pickle.load(h)[0])
assert count('births_plr_direct') == count('births_plr_v1')
PY
}

echo "[$(date '+%F %T')] final key controls started fingerprint=$RUN_FINGERPRINT"
for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"; log="$SCENE_LOG_ROOT/$scene.log"
  [[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || {
    echo "Source/config changed before $scene" >&2; exit 1;
  }
  if validate_scene "$scene"; then
    echo "[$(date '+%F %T')] [$((index+1))/100] Reusing $scene"; continue
  fi
  for stale in "$NATIVE_ROOT/${scene}_boxes.pkl" "$ONLINE_ROOT/${scene}_boxes.pkl" \
    "$DIAGNOSTICS_ROOT/${scene}.json" "$COMPONENT_ROOT/${scene}.json" \
    "$EVIDENCE_ROOT/${scene}.npz" "$log"; do
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
[[ ! -e "$ARM_ROOT/manifest.json" ]] || { echo "Key arm manifest exists" >&2; exit 1; }
"$PYTHON" "$DEV/tools/materialize_key_controls.py" \
  --scene-list "$SCENE_LIST" --components "$COMPONENT_ROOT" --output "$ARM_ROOT"
[[ ! -e "$CALR_ROOT/manifest.json" ]] || { echo "CALR manifest exists" >&2; exit 1; }
"$PYTHON" "$DEV/tools/materialize_calr_key_controls.py" \
  --scene-list "$SCENE_LIST" --baseline "$ARM_ROOT/calr_prefix" \
  --full "$ARM_ROOT/calr_full" --anchor-cache "$EVIDENCE_ROOT" \
  --output "$CALR_ROOT"

declare -A EVAL_ARMS=(
  [recar3d_key_plr_direct]="$ARM_ROOT/plr_direct"
  [recar3d_key_plr_current]="$ARM_ROOT/plr_current"
  [recar3d_key_mvsr_raw_iou_max]="$ARM_ROOT/mvsr_raw_iou_max"
  [recar3d_key_mvsr_composite_max]="$ARM_ROOT/mvsr_composite_max"
  [recar3d_key_full_raw_iou]="$ARM_ROOT/full_raw_iou"
  [recar3d_key_full_composite]="$ARM_ROOT/full_composite"
  [recar3d_key_calr_lower_threshold]="$CALR_ROOT/lower_threshold"
  [recar3d_key_calr_strongest_matched]="$CALR_ROOT/strongest_matched_budget"
  [recar3d_key_calr_full]="$CALR_ROOT/full_calr"
)
for label in $(printf '%s\n' "${!EVAL_ARMS[@]}" | sort); do
  bash "$MAIN/scripts/eval_scannet_official100_real_score.sh" \
    "$label" "${EVAL_ARMS[$label]}"
done
"$PYTHON" "$DEV/tools/summarize_key_controls.py" \
  --key-manifest "$ARM_ROOT/manifest.json" \
  --calr-manifest "$CALR_ROOT/manifest.json" \
  --eval-log-root "$EVAL_LOG_ROOT" --output "$REPORT_ROOT"
echo "[$(date '+%F %T')] final key controls complete: $REPORT_ROOT/REPORT.md"
