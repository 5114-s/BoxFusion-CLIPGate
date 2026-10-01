#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
CONFIG="$ROOT/config/scannet_t05_boxer_strict_causal_nochild_single_gpu_fps.yaml"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
OFFICIAL_SOURCE="$ROOT/reports/paper_matched_runtime_full100_20260916/official_source_06ea629"
REPORT="$ROOT/reports/original_flow_strict_causal_nochild_fps_official100_20260925"
RECEIPTS="$REPORT/receipts"
LOGS="$REPORT/logs"
MPL="$REPORT/mplconfig"

mkdir -p "$RECEIPTS/official" "$RECEIPTS/full" "$LOGS" "$MPL"
exec 9>"$REPORT/run.lock"
flock -n 9 || { echo "original-flow FPS runner is already active" >&2; exit 1; }
exec > >(tee -a "$REPORT/driver.log") 2>&1

mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 100 ]]

fingerprint() {
  sha256sum "$CONFIG" "$ROOT/demo.py" \
    "$ROOT/tools/benchmark_official_boxfusion_scene.py" \
    "$ROOT/tools/benchmark_original_flow_full_scene.py" \
    "$ROOT/tools/summarize_original_flow_fps.py" \
    "$ROOT/boxfusion/online_candidate_map.py" \
    "$ROOT/boxfusion/online_candidate_runtime.py" \
    "$OFFICIAL_SOURCE/demo.py" "$OFFICIAL_SOURCE/config/scannet.yaml" \
    | sha256sum | awk '{print $1}'
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
assert d["scene"] == scene
if arm == "official":
    assert d["schema"] == "boxfusion.paper_matched.official_scene.v1"
    assert d["consumed_raw_frames"] > 0 and d["run_seconds"] > 0
    assert d["model_load_excluded"] is True
    assert d["visualization"] is False and d["proposal_replay"] is False
else:
    assert d["arm"] == "full" and d["gpu_count"] == 1
    assert d["cuda_synchronized"] is True
    assert d["prepare_all_raw_frames"] is True
    assert d["original_frame_loop"] is True
    assert d["children_enabled"] is False
    assert d["scene_cache_prewarmed"] is False
    assert d["raw_frames"] > 0 and d["cost_seconds"] > 0
    assert d["final_output_serialization_included"] is True
    assert d["model_initialization_excluded"] is True
    assert d["visualization"] is False and d["proposal_replay"] is False
PY
}

run_official() {
  local scene="$1"
  local out="$RECEIPTS/official/$scene"
  if valid_arm official "$scene"; then
    echo "Reusing arm=official scene=$scene"
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
    PYTHONPATH="$OFFICIAL_SOURCE:$ROOT/third_party/WeDetect" \
    "$PYTHON" "$ROOT/tools/benchmark_official_boxfusion_scene.py" \
      --source-root "$OFFICIAL_SOURCE" --repository-root "$ROOT" \
      --scene "$scene" --output-root "$out" --device-index 0 \
      > "$LOGS/${scene}.official.log" 2>&1
  valid_arm official "$scene"
}

run_full() {
  local scene="$1"
  local out="$RECEIPTS/full/$scene"
  if valid_arm full "$scene"; then
    echo "Reusing arm=full scene=$scene"
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
    "$PYTHON" "$ROOT/tools/benchmark_original_flow_full_scene.py" \
      --scene "$scene" --config "$CONFIG" --output-root "$out" \
      --device-index 0 > "$LOGS/${scene}.full.log" 2>&1
  valid_arm full "$scene"
}

echo "[$(date '+%F %T')] original-flow official100 started fingerprint=$RUN_FINGERPRINT"
for index in "${!SCENES[@]}"; do
  [[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || {
    echo "Source/config changed during run" >&2
    exit 1
  }
  scene="${SCENES[$index]}"
  echo "[$(date '+%F %T')] [$((index + 1))/100] $scene"
  if (( index % 2 == 0 )); then
    run_official "$scene"
    run_full "$scene"
  else
    run_full "$scene"
    run_official "$scene"
  fi
  official_fps="$("$PYTHON" -c "import json; d=json.load(open('$RECEIPTS/official/$scene/runtime.json')); print(d['raw_fps'])")"
  full_fps="$("$PYTHON" -c "import json; d=json.load(open('$RECEIPTS/full/$scene/runtime.json')); print(d['fps'])")"
  echo "[$(date '+%F %T')] Completed $scene official_fps=$official_fps full_fps=$full_fps"
done

"$PYTHON" "$ROOT/tools/summarize_original_flow_fps.py" \
  --scene-list "$SCENE_LIST" --receipt-root "$RECEIPTS" --output "$REPORT"
echo "[$(date '+%F %T')] original-flow official100 complete"
