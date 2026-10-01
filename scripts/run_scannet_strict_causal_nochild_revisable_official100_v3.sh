#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
CONFIG="$ROOT/config/scannet_t05_boxer_strict_causal_nochild_revisable_official100_v3.yaml"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
MODEL="$ROOT/models/cutr_rgbd.pth"
CLIP_MODEL="$ROOT/models/open_clip_pytorch_model.bin"
CLASS_TXT="$ROOT/data/panoptic_categories_nomerge.txt"
NATIVE_ROOT="$ROOT/results/scannet_strict_causal_nochild_revisable_official100_v3_native"
ONLINE_ROOT="$ROOT/results/scannet_strict_causal_nochild_revisable_official100_v3"
DIAGNOSTICS_ROOT="$ROOT/diagnostics/strict_causal_nochild_revisable_official100_v3"
PROVIDER_DIAGNOSTICS_ROOT="$ROOT/diagnostics/strict_causal_nochild_revisable_official100_v3_provider"
LOG_ROOT="$ROOT/logs/scannet_strict_causal_nochild_revisable_official100_v3"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
REPORT_ROOT="$ROOT/reports/strict_causal_nochild_revisable_official100_v3_20260925"
FACTORIAL_ROOT="$ROOT/results/scannet_strict_causal_nochild_revisable_official100_v3_factorial"
MPL_ROOT="$LOG_ROOT/mplconfig"
FINGERPRINT_FILE="$LOG_ROOT/source.fingerprint"

for required in "$PYTHON" "$CONFIG" "$SCENE_LIST" "$MODEL" "$CLIP_MODEL" "$CLASS_TXT"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
[[ "$(sha256sum "$SCENE_LIST" | awk '{print $1}')" == \
  4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5 ]] || {
  echo "Official100 scene-list hash mismatch" >&2
  exit 1
}
mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 100 ]] || { echo "Expected 100 scenes" >&2; exit 1; }

mkdir -p "$NATIVE_ROOT" "$ONLINE_ROOT" "$DIAGNOSTICS_ROOT" \
  "$PROVIDER_DIAGNOSTICS_ROOT" "$SCENE_LOG_ROOT" "$MPL_ROOT" "$REPORT_ROOT"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "Official100 runner is already active" >&2; exit 1; }
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

fingerprint() {
  sha256sum "$CONFIG" "$ROOT/demo.py" \
    "$ROOT/boxfusion/online_candidate_map.py" \
    "$ROOT/boxfusion/online_candidate_runtime.py" \
    "$ROOT/boxfusion/box_manager.py" "$ROOT/boxfusion/instances.py" \
    "$MODEL" "$CLIP_MODEL" "$CLASS_TXT" | sha256sum | awk '{print $1}'
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
assert row["asynchronous"] is True
assert row["state"]["m1p"]["use_children"] is False
assert row["state"]["terminal_native_map_readout"] is True
assert row["state"]["uses_terminal_map_for_inference"] is False
counts = row["state"]["terminal_counts"]
assert counts["total"] == counts["native"] + counts["m1p"] + counts["m1a"]
PY
}

echo "[$(date '+%F %T')] strict-causal no-child official100 started fingerprint=$RUN_FINGERPRINT"
for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"
  log="$SCENE_LOG_ROOT/${scene}.log"
  [[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || {
    echo "Source/config changed before $scene; refusing a mixed run" >&2
    exit 1
  }
  if validate_scene "$scene"; then
    echo "[$(date '+%F %T')] [$((index + 1))/100] Reusing $scene"
    continue
  fi
  for stale in "$NATIVE_ROOT/${scene}_boxes.pkl" "$ONLINE_ROOT/${scene}_boxes.pkl" \
    "$DIAGNOSTICS_ROOT/${scene}.json" "$log"; do
    [[ ! -e "$stale" ]] || { echo "Invalid partial artifact: $stale" >&2; exit 1; }
  done
  echo "[$(date '+%F %T')] [$((index + 1))/100] Running $scene"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES=0,1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
    MPLCONFIGDIR="$MPL_ROOT" \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
    PYTHONPATH="$ROOT:$ROOT/third_party/WeDetect" \
    "$PYTHON" demo.py scannet \
      --model-path "$MODEL" --clip_path "$CLIP_MODEL" \
      --class_txt "$CLASS_TXT" --config "$CONFIG" \
      --device cuda:0 --seq "$scene"
  ) > "$log" 2>&1
  validate_scene "$scene" || { echo "Scene validation failed: $scene; see $log" >&2; exit 1; }
  echo "[$(date '+%F %T')] [$((index + 1))/100] Completed $scene $(grep -F 'Online runtime receipt |' "$log")"
done

[[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || {
  echo "Source/config changed during the run" >&2
  exit 1
}
"$PYTHON" "$ROOT/tools/summarize_online_candidate_official100.py" \
  --scene-list "$SCENE_LIST" --logs "$SCENE_LOG_ROOT" \
  --diagnostics "$DIAGNOSTICS_ROOT" --predictions "$ONLINE_ROOT" \
  --output "$REPORT_ROOT"

if [[ -e "$FACTORIAL_ROOT/manifest.json" ]]; then
  echo "Factorial manifest already exists; refusing to overwrite" >&2
  exit 1
fi
"$PYTHON" "$ROOT/tools/materialize_online_nochild_factorial.py" \
  --scene-list "$SCENE_LIST" --native "$NATIVE_ROOT" --full "$ONLINE_ROOT" \
  --diagnostics "$DIAGNOSTICS_ROOT" --output "$FACTORIAL_ROOT"

for arm in base p a p_a m2 p_m2 a_m2 p_a_m2; do
  echo "[$(date '+%F %T')] Evaluating paired arm=$arm"
  bash "$ROOT/scripts/eval_scannet_official100_real_score.sh" \
    "strict_causal_nochild_revisable_official100_v3_${arm}" "$FACTORIAL_ROOT/$arm"
done
echo "[$(date '+%F %T')] strict-causal no-child official100 complete"
