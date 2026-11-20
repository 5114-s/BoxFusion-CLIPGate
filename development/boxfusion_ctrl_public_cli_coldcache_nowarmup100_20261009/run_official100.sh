#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
DEV="$ROOT/development/boxfusion_ctrl_public_cli_coldcache_nowarmup100_20261009"
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
SOURCE="$ROOT/development/official_github_exact_paired_efficiency_20261009/recar_source"
CONFIG="$SOURCE/config/scannet_recar3d_final_novis.yaml"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
REFERENCE="$ROOT/reports/official_github_boxfusion_vs_final_recar_public_cli_coldcache_nowarmup100_20261009/receipts/recar"
REPORT="$ROOT/reports/boxfusion_ctrl_public_cli_coldcache_nowarmup100_20261009"
RECEIPTS="$REPORT/receipts/ctrl"
LOGS="$REPORT/logs"
MPL="$REPORT/mplconfig"
GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', *' '$1 == 0 {print $2}')"

mkdir -p "$RECEIPTS" "$LOGS" "$MPL"
exec 9>"$REPORT/run.lock"
flock -n 9 || { echo "BoxFusion-Ctrl FPS runner is already active" >&2; exit 1; }
exec > >(tee -a "$REPORT/driver.log") 2>&1

mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 100 ]]

fingerprint() {
  find "$DEV/tools" "$SOURCE" -type f \
    ! -path '*/__pycache__/*' ! -path '*/.pytest_cache/*' \
    ! -name '*.pyc' -print0 \
    | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
}

RUN_FINGERPRINT="$(fingerprint)"
printf '%s\n' "$RUN_FINGERPRINT" > "$REPORT/source.fingerprint"
sha256sum "$SCENE_LIST" > "$REPORT/scene_list.sha256"
{
  echo "created=$(date --iso-8601=seconds)"
  echo "route=BoxFusion-Ctrl"
  echo "source=$SOURCE"
  echo "source_parent_commit=ad52642cc6d628ba0f52208181230c33a152c77e"
  echo "active_boxer=true"
  echo "reliable_view_top_k=3"
  echo "online_candidate_map=false"
  echo "protocol=public_demo_native_wall_clock_verified_cold_scene_files"
  echo "gpu_visible=0"
  echo "visualization=false"
  echo "cuda_synchronization=false"
  echo "synthetic_model_warmup=false"
  echo "public_cli_viz_on_gt_points=true"
  sha256sum "$SOURCE/demo.py" "$SOURCE/boxfusion/boxer_lifter.py" \
    "$SOURCE/boxfusion/reliable_views.py" "$CONFIG" \
    "$DEV/tools/benchmark_ctrl_scene.py"
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
} > "$REPORT/environment.txt"

valid_scene() {
  local scene="$1"
  local receipt="$RECEIPTS/$scene/runtime.json"
  [[ -s "$receipt" ]] || return 1
  "$PYTHON" - "$receipt" "$scene" <<'PY' >/dev/null
import json, sys
p, scene = sys.argv[1:]
d = json.load(open(p, encoding="utf-8"))
assert d["arm"] == "ctrl" and d["scene"] == scene
assert d["gpu_count"] == 1 and d["gap"] == 25
assert d["score_threshold"] == 0.5
assert d["timing_protocol"] == "public_demo_native_time_time"
assert d["native_timer_call_count"] == 2
assert d["cold_page_cache_verified"] is True
assert d["resident_bytes_at_timing_boundary"] == 0
assert d["cuda_synchronized"] is False
assert d["prepare_all_raw_frames"] is False
assert d["model_initialization_excluded"] is True
assert d["visualization"] is False
assert d["proposal_replay"] is False
assert d["synthetic_model_warmup"] is False
assert d["viz_on_gt_points"] is True
assert d["online_candidate_map_enabled"] is False
assert d["active_boxer"] is True
assert d["reliable_view_top_k"] == 3
assert d["raw_frames"] > 0 and d["cost_seconds"] > 0
PY
}

wait_for_gpu0_idle() {
  while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
      | grep -Fxq "$GPU_UUID"; do
    echo "[$(date '+%F %T')] Waiting for exclusive GPU 0 access"
    sleep 60
  done
}

gpu0_temperature() {
  nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits \
    | awk -F', *' '$1 == 0 {print $2}'
}

wait_for_initial_temperature() {
  local current
  wait_for_gpu0_idle
  while true; do
    current="$(gpu0_temperature)"
    echo "[$(date '+%F %T')] GPU 0 temperature=${current}C target<=32C"
    (( current <= 32 )) && return
    sleep 30
  done
}

validate_first_scene_route() {
  local scene="$1"
  local candidate="$RECEIPTS/$scene/native/${scene}_boxes.pkl"
  local reference="$REFERENCE/$scene/native/${scene}_boxes.pkl"
  "$PYTHON" - "$candidate" "$reference" "$REPORT/route_validation.json" <<'PY'
import json, pickle, sys
import numpy as np
candidate_path, reference_path, output_path = sys.argv[1:]
candidate = pickle.load(open(candidate_path, "rb"))[0]
reference = pickle.load(open(reference_path, "rb"))[0]
assert len(candidate) == len(reference)
assert [row[0] for row in candidate] == [row[0] for row in reference]
box_diff = max(
    (float(np.max(np.abs(a[1] - b[1]))) for a, b in zip(candidate, reference)),
    default=0.0,
)
score_diff = max(
    (abs(float(a[2]) - float(b[2])) for a, b in zip(candidate, reference)),
    default=0.0,
)
assert box_diff <= 0.02, box_diff
assert score_diff <= 0.01, score_diff
result = {
    "schema": "boxfusion.ctrl.route_validation.v1",
    "candidate": candidate_path,
    "reference_full_run_native": reference_path,
    "records": len(candidate),
    "labels_and_order_equal": True,
    "max_abs_corner_difference_m": box_diff,
    "max_abs_score_difference": score_diff,
    "tolerances": {"corner_m": 0.02, "score": 0.01},
    "passed": True,
}
open(output_path, "w", encoding="utf-8").write(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(result, sort_keys=True))
PY
}

wait_for_initial_temperature
echo "[$(date '+%F %T')] BoxFusion-Ctrl official100 FPS run started fingerprint=$RUN_FINGERPRINT"
for index in "${!SCENES[@]}"; do
  [[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || {
    echo "Frozen benchmark source changed during run" >&2
    exit 1
  }
  scene="${SCENES[$index]}"
  if valid_scene "$scene"; then
    echo "[$(date '+%F %T')] Reusing ctrl [$((index + 1))/100] $scene"
  else
    out="$RECEIPTS/$scene"
    if [[ -e "$out/runtime.json" ]]; then
      stamp="$(date '+%Y%m%d_%H%M%S')"
      mkdir -p "$REPORT/interrupted/$scene/$stamp"
      mv "$out" "$REPORT/interrupted/$scene/$stamp/ctrl"
    fi
    mkdir -p "$out"
    wait_for_gpu0_idle
    echo "[$(date '+%F %T')] Running ctrl [$((index + 1))/100] $scene"
    CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
      MPLCONFIGDIR="$MPL" \
      LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
      PYTHONPATH="$SOURCE:$ROOT/third_party/WeDetect" \
      "$PYTHON" "$DEV/tools/benchmark_ctrl_scene.py" \
        --arm ctrl --scene "$scene" --source-root "$SOURCE" \
        --repository-root "$ROOT" --output-root "$out" \
        --config "$CONFIG" --device-index 0 \
        > "$LOGS/${scene}.ctrl.log" 2>&1
    valid_scene "$scene"
  fi
  if (( index == 0 )); then
    validate_first_scene_route "$scene"
  fi
  fps="$($PYTHON -c "import json; print(json.load(open('$RECEIPTS/$scene/runtime.json'))['fps'])")"
  echo "[$(date '+%F %T')] Completed ctrl [$((index + 1))/100] $scene fps=$fps"
done

"$PYTHON" "$DEV/tools/summarize_ctrl.py" \
  --scene-list "$SCENE_LIST" --receipt-root "$REPORT/receipts" \
  --arm ctrl --output "$REPORT/summary.json"
echo "[$(date '+%F %T')] BoxFusion-Ctrl official100 FPS run complete"
