#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
DEV="$ROOT/development/paper_native_sequential_coldcache_nowarmup_efficiency_20261009"
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
ORIGINAL="$ROOT/development/official_github_exact_paired_efficiency_20261009/official_source_b2e0219"
RECAR="$ROOT/development/official_github_exact_paired_efficiency_20261009/recar_source"
CONFIG="$RECAR/config/scannet_recar3d_final_novis.yaml"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
REPORT="$ROOT/reports/official_github_boxfusion_vs_final_recar_paper_native_sequential_coldcache_nowarmup100_20261009"
RECEIPTS="$REPORT/receipts"
LOGS="$REPORT/logs"
MPL="$REPORT/mplconfig"
GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', *' '$1 == 0 {print $2}')"

mkdir -p "$RECEIPTS/original" "$RECEIPTS/recar" "$LOGS" "$MPL"
exec 9>"$REPORT/run.lock"
flock -n 9 || { echo "paired efficiency runner is already active" >&2; exit 1; }
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
  echo "protocol=sequential_arms_public_demo_native_wall_clock_verified_cold_scene_files"
  echo "phase_order=recar_then_original"
  echo "original_source=$ORIGINAL"
  echo "original_repository=https://github.com/lanlan96/BoxFusion.git"
  echo "original_commit=b2e0219a7284249bad4a4a8925066839fe2fa33b"
  echo "original_source_kind=clean_git_archive"
  echo "recar_source=$RECAR"
  echo "recar_parent_git_commit=ad52642cc6d628ba0f52208181230c33a152c77e"
  echo "recar_route=PLR-v1+CALR-v1+MVSR-no_visibility-Max"
  echo "recar_reference_run=$ROOT/reports/final_novis_calrv1_recar_fps_reproduction100_20261008"
  echo "visualization=false"
  echo "cuda_synchronization=false"
  echo "output_serialization_in_timed_interval=false"
  echo "os_page_cache_control=POSIX_FADV_DONTNEED_then_fincore_zero_resident_bytes"
  echo "synthetic_model_warmup=false"
  sha256sum "$RECAR/demo.py" \
    "$RECAR/boxfusion/reliability_candidate_map.py" \
    "$RECAR/boxfusion/native_reliability_reranker.py" \
    "$RECAR/boxfusion/m1_anchor_online.py" \
    "$CONFIG"
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
} > "$REPORT/environment.txt"

valid_arm() {
  local arm="$1"
  local scene="$2"
  local path="$RECEIPTS/$arm/$scene/runtime.json"
  [[ -s "$path" ]] || return 1
  "$PYTHON" - "$path" "$arm" "$scene" "$RECEIPTS" <<'PY' >/dev/null
import json, pathlib, sys
p, arm, scene, receipts = sys.argv[1:]
d = json.load(open(p, encoding="utf-8"))
assert d["arm"] == arm and d["scene"] == scene
assert d["gpu_count"] == 1 and d["gap"] == 25
assert d["score_threshold"] == 0.5
assert d["timing_protocol"] == "public_demo_native_time_time"
assert d["native_timer_call_count"] == 2
assert d["cold_page_cache_verified"] is True
assert d["cache_eviction_method"] == "POSIX_FADV_DONTNEED_then_fincore"
assert d["cache_evicted_files"] > 0 and d["cache_evicted_bytes"] > 0
assert d["resident_bytes_at_timing_boundary"] == 0
assert d["cuda_synchronized"] is False
assert d["prepare_all_raw_frames"] is False
assert d["scene_cache_prewarmed"] is False
assert d["synthetic_model_warmup"] is False
assert d["os_page_cache_control"] == "verified_cold_scene_files"
assert d["model_initialization_excluded"] is True
assert d["visualization"] is False and d["proposal_replay"] is False
assert d["final_output_serialization_included"] is False
assert d["raw_frames"] > 0 and d["cost_seconds"] > 0 and d["fps"] > 0
if arm == "recar":
    diag_path = pathlib.Path(receipts) / arm / scene / "online_diagnostics" / f"{scene}.json"
    q = json.load(open(diag_path, encoding="utf-8"))
    s = q["state"]
    assert q["strictly_causal"] is True and s["strictly_causal"] is True
    assert s["selected_calr_v1"] is True
    assert s["m1a"]["frames_seen"] == 0
    assert s["m1a_control"]["frames_seen"] > 0
    assert s["m1p"]["use_children"] is False
    assert s["m2"]["strength_variant"] == "no_visibility"
    assert s["m2"]["support_mode"] == "max"
    assert not s["m2_factor_controls"] and s["raw_m2_control"] is None
    assert not s["plr_controls"] and s["direct_plr_control"] is None
    assert q["provider"]["synthetic_warmup"] is False
    assert q["provider"]["warmup_seconds_before_stream"] == 0.0
    assert q["pending_at_close"] == 0
PY
}

wait_for_gpu0_idle() {
  while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
      | grep -Fxq "$GPU_UUID"; do
    echo "[$(date '+%F %T')] Waiting for exclusive GPU 0 access"
    sleep 60
  done
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
  if [[ -e "$out/runtime.json" ]]; then
    local stamp archive
    stamp="$(date '+%Y%m%d_%H%M%S')"
    archive="$REPORT/interrupted/$scene/$arm/$stamp"
    mkdir -p "$(dirname "$archive")"
    mv "$out" "$archive"
    if [[ -e "$LOGS/${scene}.${arm}.log" ]]; then
      mv "$LOGS/${scene}.${arm}.log" "$archive.log"
    fi
    echo "Archived invalid partial receipt to $archive"
  fi
  mkdir -p "$out"
  wait_for_gpu0_idle
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

gpu0_temperature() {
  nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits \
    | awk -F', *' '$1 == 0 {print $2}'
}

wait_for_thermal_recovery() {
  local target="$1"
  local current
  wait_for_gpu0_idle
  while true; do
    current="$(gpu0_temperature)"
    echo "[$(date '+%F %T')] GPU 0 cooldown current=${current}C target<=${target}C"
    (( current <= target )) && return
    sleep 30
  done
}

run_phase() {
  local arm="$1"
  echo "[$(date '+%F %T')] phase=$arm started"
  for index in "${!SCENES[@]}"; do
    [[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || {
      echo "Frozen benchmark source changed during run" >&2
      exit 1
    }
    scene="${SCENES[$index]}"
    echo "[$(date '+%F %T')] phase=$arm [$((index + 1))/100] $scene"
    run_arm "$arm" "$scene"
    arm_fps="$("$PYTHON" -c "import json; print(json.load(open('$RECEIPTS/$arm/$scene/runtime.json'))['fps'])")"
    echo "[$(date '+%F %T')] Completed arm=$arm scene=$scene fps=$arm_fps"
  done
  "$PYTHON" "$DEV/tools/summarize_single_arm.py" \
    --scene-list "$SCENE_LIST" --receipt-root "$RECEIPTS" \
    --arm "$arm" --output "$REPORT/${arm}_phase_summary.json"
  echo "[$(date '+%F %T')] phase=$arm complete"
}

INITIAL_MAX_TEMPERATURE_C=32
wait_for_thermal_recovery "$INITIAL_MAX_TEMPERATURE_C"
BASELINE_TEMPERATURE_C="$(gpu0_temperature)"
THERMAL_TARGET_C=$((BASELINE_TEMPERATURE_C + 2))
{
  echo "initial_phase_max_temperature_c=$INITIAL_MAX_TEMPERATURE_C"
  echo "baseline_gpu0_temperature_c=$BASELINE_TEMPERATURE_C"
  echo "phase_transition_target_temperature_c=$THERMAL_TARGET_C"
} > "$REPORT/thermal_protocol.txt"

echo "[$(date '+%F %T')] paper-native sequential verified-cold-cache official100 started fingerprint=$RUN_FINGERPRINT"
run_phase recar
wait_for_thermal_recovery "$THERMAL_TARGET_C"
echo "phase2_start_gpu0_temperature_c=$(gpu0_temperature)" >> "$REPORT/thermal_protocol.txt"
run_phase original

"$PYTHON" "$DEV/tools/summarize.py" \
  --scene-list "$SCENE_LIST" --receipt-root "$RECEIPTS" --output "$REPORT"
echo "[$(date '+%F %T')] paper-native sequential verified-cold-cache official100 complete"
