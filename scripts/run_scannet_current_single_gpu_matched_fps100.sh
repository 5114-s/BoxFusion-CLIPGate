#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
CONFIG="$ROOT/config/scannet_t05_boxer_strict_causal_nochild_single_gpu_fps.yaml"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
REPORT="$ROOT/reports/current_single_gpu_matched_fps_official100_20260924_v2"
RECEIPTS="$REPORT/receipts"
LOGS="$REPORT/logs"
MPL="$REPORT/mplconfig"

mkdir -p "$RECEIPTS/control" "$RECEIPTS/full" "$LOGS" "$MPL"
exec 9>"$REPORT/run.lock"
flock -n 9 || { echo "matched FPS runner is already active" >&2; exit 1; }
exec > >(tee -a "$REPORT/driver.log") 2>&1

mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 100 ]]

fingerprint() {
  sha256sum "$CONFIG" "$ROOT/demo.py" \
    "$ROOT/tools/benchmark_current_boxfusion_scene.py" \
    "$ROOT/tools/summarize_current_single_gpu_matched_fps.py" \
    "$ROOT/boxfusion/online_candidate_map.py" \
    "$ROOT/boxfusion/online_candidate_runtime.py" | sha256sum | awk '{print $1}'
}

RUN_FINGERPRINT="$(fingerprint)"
if [[ -s "$REPORT/source.fingerprint" ]]; then
  [[ "$(tr -d '\n' < "$REPORT/source.fingerprint")" == "$RUN_FINGERPRINT" ]] || {
    echo "Source/config fingerprint differs from existing run" >&2
    exit 1
  }
else
  printf '%s\n' "$RUN_FINGERPRINT" > "$REPORT/source.fingerprint"
fi

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
assert d["gpu_count"] == 1
assert d["cuda_synchronized"] is True
assert d["scene_cache_prewarmed"] is True
assert d["final_output_serialization_included"] is True
assert d["model_initialization_excluded"] is True
assert d["visualization"] is False and d["proposal_replay"] is False
assert d["raw_frames"] > 0 and d["cost_seconds"] > 0 and d["fps"] > 0
if arm == "full":
    assert d["children_enabled"] is False
PY
}

run_arm() {
  local arm="$1"
  local scene="$2"
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
    PYTHONPATH="$ROOT:$ROOT/third_party/WeDetect" \
    "$PYTHON" "$ROOT/tools/benchmark_current_boxfusion_scene.py" \
      --arm "$arm" --scene "$scene" --config "$CONFIG" \
      --output-root "$out" --device-index 0 \
      > "$LOGS/${scene}.${arm}.log" 2>&1
  valid_arm "$arm" "$scene"
}

echo "[$(date '+%F %T')] current-code single-GPU matched official100 started fingerprint=$RUN_FINGERPRINT"
for index in "${!SCENES[@]}"; do
  [[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || {
    echo "Source/config changed during run" >&2
    exit 1
  }
  scene="${SCENES[$index]}"
  echo "[$(date '+%F %T')] [$((index + 1))/100] $scene"
  if (( index % 2 == 0 )); then
    run_arm control "$scene"
    run_arm full "$scene"
  else
    run_arm full "$scene"
    run_arm control "$scene"
  fi
  control_fps="$("$PYTHON" -c "import json; print(json.load(open('$RECEIPTS/control/$scene/runtime.json'))['fps'])")"
  full_fps="$("$PYTHON" -c "import json; print(json.load(open('$RECEIPTS/full/$scene/runtime.json'))['fps'])")"
  echo "[$(date '+%F %T')] Completed $scene control_fps=$control_fps full_fps=$full_fps"
done

"$PYTHON" "$ROOT/tools/summarize_current_single_gpu_matched_fps.py" \
  --scene-list "$SCENE_LIST" --receipt-root "$RECEIPTS" --output "$REPORT"
echo "[$(date '+%F %T')] current-code single-GPU matched official100 complete"
