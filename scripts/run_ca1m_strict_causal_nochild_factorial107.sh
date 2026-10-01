#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
CONFIG="$ROOT/config/ca1m_thr15_boxer_strict_causal_nochild_full107.yaml"
SCENE_LIST="$ROOT/tools/boxfusion_tr3d_pipeline/evaluation/data_util/meta_data/ca1m_val_full107.txt"
MODEL="$ROOT/models/cutr_rgbd.pth"
CLIP_MODEL="$ROOT/models/open_clip_pytorch_model.bin"
CLASS_TXT="$ROOT/data/panoptic_categories_nomerge.txt"
DATA_ROOT=/extra/ZhaoX/boxfusion_ca1m
NATIVE_ROOT="$ROOT/results/ca1m_strict_causal_nochild_full107_native"
ONLINE_ROOT="$ROOT/results/ca1m_strict_causal_nochild_full107"
DIAGNOSTICS_ROOT="$ROOT/diagnostics/ca1m_strict_causal_nochild_full107"
LOG_ROOT="$ROOT/logs/ca1m_strict_causal_nochild_full107"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
FACTORIAL_ROOT="$ROOT/results/ca1m_strict_causal_nochild_full107_factorial"
REPORT_ROOT="$ROOT/reports/ca1m_strict_causal_nochild_factorial_20260925"
MPL_ROOT="$LOG_ROOT/mplconfig"
FINGERPRINT_FILE="$LOG_ROOT/source.fingerprint"

for required in "$PYTHON" "$CONFIG" "$SCENE_LIST" "$MODEL" "$CLIP_MODEL" "$CLASS_TXT"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
[[ "$(sha256sum "$SCENE_LIST" | awk '{print $1}')" == \
  bd5f3fc66168114048a1b12addc45949c8f54f9c016b921bacfb6fe9e3e7dc2f ]] || {
  echo "CA-1M-107 scene-list hash mismatch" >&2
  exit 1
}
mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 107 ]] || { echo "Expected 107 scenes" >&2; exit 1; }

mkdir -p "$NATIVE_ROOT" "$ONLINE_ROOT" "$DIAGNOSTICS_ROOT" \
  "$SCENE_LOG_ROOT" "$MPL_ROOT" "$REPORT_ROOT"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "CA-1M strict-causal runner is already active" >&2; exit 1; }
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

fingerprint() {
  sha256sum "$CONFIG" "$ROOT/demo.py" \
    "$ROOT/boxfusion/online_candidate_map.py" \
    "$ROOT/boxfusion/online_candidate_runtime.py" \
    "$ROOT/boxfusion/box_manager.py" "$ROOT/boxfusion/instances.py" \
    "$ROOT/tools/materialize_online_nochild_factorial.py" \
    "$ROOT/tools/evaluate_ca1m_online_factorial.py" \
    "$MODEL" "$CLIP_MODEL" "$CLASS_TXT" "$SCENE_LIST" \
    | sha256sum | awk '{print $1}'
}
RUN_FINGERPRINT="$(fingerprint)"
if [[ -s "$FINGERPRINT_FILE" ]]; then
  [[ "$(tr -d '\n' < "$FINGERPRINT_FILE")" == "$RUN_FINGERPRINT" ]] || {
    echo "Source/config fingerprint differs from the existing run" >&2
    exit 1
  }
else
  printf '%s\n' "$RUN_FINGERPRINT" > "$FINGERPRINT_FILE"
fi

validate_scene() {
  local scene="$1" log="$SCENE_LOG_ROOT/$1.log"
  [[ -s "$NATIVE_ROOT/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$ONLINE_ROOT/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$DIAGNOSTICS_ROOT/${scene}.json" ]] || return 1
  [[ -s "$log" ]] || return 1
  [[ "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]] || return 1
  [[ "$(grep -Fc 'Online candidate map summary | causal=True scene_end_inference=False' "$log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|RuntimeError|CUDA out of memory' "$log" || return 1
  "$PYTHON" - "$DIAGNOSTICS_ROOT/${scene}.json" <<'PY' >/dev/null
import json, sys
row = json.load(open(sys.argv[1], encoding="utf-8"))
assert row["strictly_causal"] is True
assert row["online_incremental"] is True
assert row["scene_end_inference"] is False
assert row["scene_end_assembly"] is True
assert row["state"]["m1p"]["use_children"] is False
assert row["state"]["terminal_native_map_readout"] is True
assert row["state"]["uses_terminal_map_for_inference"] is False
counts = row["state"]["terminal_counts"]
assert counts["total"] == counts["native"] + counts["m1p"] + counts["m1a"]
PY
}

echo "[$(date '+%F %T')] CA-1M strict-causal no-child factorial107 started fingerprint=$RUN_FINGERPRINT"
for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"
  log="$SCENE_LOG_ROOT/${scene}.log"
  [[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || {
    echo "Source/config changed before $scene; refusing a mixed run" >&2
    exit 1
  }
  if validate_scene "$scene"; then
    echo "[$(date '+%F %T')] [$((index + 1))/107] Reusing $scene"
    continue
  fi
  for stale in "$NATIVE_ROOT/${scene}_boxes.pkl" "$ONLINE_ROOT/${scene}_boxes.pkl" \
    "$DIAGNOSTICS_ROOT/${scene}.json" "$log"; do
    [[ ! -e "$stale" ]] || { echo "Invalid partial artifact: $stale" >&2; exit 1; }
  done
  echo "[$(date '+%F %T')] [$((index + 1))/107] Running $scene on physical GPU 1"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
    MPLCONFIGDIR="$MPL_ROOT" \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
    PYTHONPATH="$ROOT:$ROOT/third_party/WeDetect" \
    "$PYTHON" demo.py CA1M \
      --model-path "$MODEL" --clip_path "$CLIP_MODEL" \
      --class_txt "$CLASS_TXT" --config "$CONFIG" \
      --device cuda:0 --seq "$scene"
  ) > "$log" 2>&1
  validate_scene "$scene" || { echo "Scene validation failed: $scene; see $log" >&2; exit 1; }
  echo "[$(date '+%F %T')] [$((index + 1))/107] Completed $scene $(grep -F 'Online runtime receipt |' "$log")"
done

[[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || {
  echo "Source/config changed during the run" >&2
  exit 1
}
if [[ -e "$FACTORIAL_ROOT/manifest.json" ]]; then
  echo "Factorial manifest already exists; refusing to overwrite" >&2
  exit 1
fi
"$PYTHON" "$ROOT/tools/materialize_online_nochild_factorial.py" \
  --scene-list "$SCENE_LIST" --expected-scenes 107 \
  --native "$NATIVE_ROOT" --full "$ONLINE_ROOT" \
  --diagnostics "$DIAGNOSTICS_ROOT" --output "$FACTORIAL_ROOT"

"$PYTHON" "$ROOT/tools/evaluate_ca1m_online_factorial.py" \
  --data-root "$DATA_ROOT" --scene-list "$SCENE_LIST" \
  --factorial-root "$FACTORIAL_ROOT" --output "$REPORT_ROOT" \
  2>&1 | tee "$LOG_ROOT/evaluate_factorial.log"
sha256sum "$CONFIG" "$SCENE_LIST" "$FACTORIAL_ROOT/manifest.json" \
  "$REPORT_ROOT/results.json" > "$REPORT_ROOT/artifact_sha256.txt"
echo "[$(date '+%F %T')] CA-1M strict-causal no-child factorial107 complete"
