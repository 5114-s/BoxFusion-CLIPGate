#!/usr/bin/env bash
set -euo pipefail

# ReCaR-3D locked route (PLR-v1 + CALR-v1 + MVSR-v2 max, strictly causal,
# online, no children) evaluated on the pre-registered extra50 holdout
# scenes.  Every method constant is identical to the frozen
# reliability-v2 official100 screening run; only output/input paths and the
# scene list differ.  Resumable per scene; refuses partial artifacts.

DEV=/data/ZhaoX/BoxFusion/development/reliability_v2
MAIN=/data/ZhaoX/BoxFusion
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
CONFIG="$DEV/config/scannet_t05_boxer_reliability_v2_extra50.yaml"
SCENE_LIST="$MAIN/evaluation/data_util/meta_data/scannetv2_val_f0_extra50.txt"
MODEL="$MAIN/models/cutr_rgbd.pth"
CLIP_MODEL="$MAIN/models/open_clip_pytorch_model.bin"
CLASS_TXT="$MAIN/data/panoptic_categories_nomerge.txt"
NATIVE_ROOT="$DEV/results/scannet_reliability_v2_extra50_native"
ONLINE_ROOT="$DEV/results/scannet_reliability_v2_extra50"
DIAGNOSTICS_ROOT="$DEV/diagnostics/scannet_reliability_v2_extra50"
COMPONENT_ROOT="$DEV/results/scannet_reliability_v2_extra50_components"
FACTORIAL_ROOT="$DEV/results/scannet_recar3d_extra50_factorial"
REPORT_ROOT="$MAIN/reports/recar3d_extra50_20260928"
LOG_ROOT="$DEV/logs/scannet_reliability_v2_extra50"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
MPL_ROOT="$LOG_ROOT/mplconfig"
FINGERPRINT_FILE="$LOG_ROOT/source.fingerprint"

for required in "$PYTHON" "$CONFIG" "$SCENE_LIST" "$MODEL" "$CLIP_MODEL" "$CLASS_TXT"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
[[ "$(sha256sum "$SCENE_LIST" | awk '{print $1}')" == \
  f3820d818cbe5e8105a57fb758870ff6aa02b305f22e19d3019e1c701952b1ea ]] || {
  echo "Extra50 scene-list hash mismatch" >&2; exit 1;
}
mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 50 ]] || { echo "Expected 50 scenes" >&2; exit 1; }
mkdir -p "$NATIVE_ROOT" "$ONLINE_ROOT" "$DIAGNOSTICS_ROOT" "$COMPONENT_ROOT" \
  "$SCENE_LOG_ROOT" "$MPL_ROOT" "$REPORT_ROOT"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "Extra50 runner is active" >&2; exit 1; }
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

fingerprint() {
  sha256sum "$CONFIG" "$DEV/demo.py" \
    "$DEV/scripts/run_reliability_v2_extra50.sh" \
    "$DEV/tools/materialize_recar3d_extra50_factorial.py" \
    "$DEV/tools/summarize_recar3d_extra50_factorial.py" \
    "$DEV/scripts/eval_scannet_extra50_real_score.sh" \
    "$DEV/boxfusion/causal_reliability.py" \
    "$DEV/boxfusion/native_reliability_reranker.py" \
    "$DEV/boxfusion/m1_anchor_reliability_online.py" \
    "$DEV/boxfusion/reliability_candidate_map.py" \
    "$DEV/boxfusion/m1_anchor_online.py" \
    "$DEV/boxfusion/online_candidate_map.py" \
    "$DEV/boxfusion/online_candidate_runtime.py" \
    "$DEV/boxfusion/box_manager.py" "$DEV/boxfusion/instances.py" \
    "$MODEL" "$CLIP_MODEL" "$CLASS_TXT" | sha256sum | awk '{print $1}'
}
RUN_FINGERPRINT="$(fingerprint)"
if [[ -s "$FINGERPRINT_FILE" ]]; then
  [[ "$(tr -d '\n' < "$FINGERPRINT_FILE")" == "$RUN_FINGERPRINT" ]] || {
    echo "Source/config fingerprint differs from existing extra50 run" >&2; exit 1;
  }
else
  printf '%s\n' "$RUN_FINGERPRINT" > "$FINGERPRINT_FILE"
fi

COMPONENTS=(native_native native_first native_mean native_max native_ema \
  native_diverse_max native_reliability births_plr_v1 births_calr_v1 births_calr_v2)

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
assert diagnostic['strictly_causal'] is True
assert diagnostic['online_incremental'] is True
assert diagnostic['scene_end_inference'] is False
assert diagnostic['state']['cross_branch_dedup'] is False
assert diagnostic['state']['m1p']['use_children'] is False
assert diagnostic['state']['m1a_control'] is not None
assert diagnostic['state']['m2']['support_mode'] == 'reliability'
assert components['single_online_pass'] is True
assert components['future_frames_used'] is False
assert len(components['counts']) == 10
PY
}

echo "[$(date '+%F %T')] Waiting for two idle GPUs"
while true; do
  if values="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
    if awk -F, '
      BEGIN { ok=1; n=0 }
      { gsub(/ /, "", $1); gsub(/ /, "", $2); n++; if ($1 > 1500 || $2 > 10) ok=0 }
      END { exit !(n >= 2 && ok) }
    ' <<<"$values"; then
      break
    fi
  fi
  sleep 60
done
echo "[$(date '+%F %T')] GPUs idle; extra50 run started fingerprint=$RUN_FINGERPRINT"

for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"; log="$SCENE_LOG_ROOT/$scene.log"
  [[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || {
    echo "Source/config changed before $scene" >&2; exit 1;
  }
  if validate_scene "$scene"; then
    echo "[$(date '+%F %T')] [$((index+1))/50] Reusing $scene"; continue
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
  echo "[$(date '+%F %T')] [$((index+1))/50] Running $scene"
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
  echo "[$(date '+%F %T')] [$((index+1))/50] Completed $scene"
done

[[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || { echo "Source changed" >&2; exit 1; }

if [[ ! -e "$FACTORIAL_ROOT/manifest.json" ]]; then
  "$PYTHON" "$DEV/tools/materialize_recar3d_extra50_factorial.py" \
    --scene-list "$SCENE_LIST" --components "$COMPONENT_ROOT" \
    --output "$FACTORIAL_ROOT"
fi

ARMS=(base plr calr mvsr plr_calr plr_mvsr calr_mvsr full)
for arm in "${ARMS[@]}"; do
  log="$MAIN/logs/scannet_extra50_real_score/recar3d_extra50_factorial_${arm}.log"
  if [[ -s "$log" ]] && [[ "$(grep -Fc 'eval mAP:' "$log")" -eq 3 ]]; then
    echo "[$(date '+%F %T')] Eval cached: $arm"; continue
  fi
  bash "$DEV/scripts/eval_scannet_extra50_real_score.sh" \
    "recar3d_extra50_factorial_${arm}" "$FACTORIAL_ROOT/$arm"
done

"$PYTHON" "$DEV/tools/summarize_recar3d_extra50_factorial.py" \
  --manifest "$FACTORIAL_ROOT/manifest.json" \
  --log-root "$MAIN/logs/scannet_extra50_real_score" \
  --output "$REPORT_ROOT"
cp "$SCENE_LIST" "$REPORT_ROOT/scenes.txt"
cp "$FINGERPRINT_FILE" "$REPORT_ROOT/source.fingerprint"
echo "[$(date '+%F %T')] extra50 factorial complete: $REPORT_ROOT/REPORT.md"
