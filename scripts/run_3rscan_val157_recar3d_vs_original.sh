#!/usr/bin/env bash
set -euo pipefail

# 3RScan full official validation split (157 scans) comparison: locked ReCaR-3D route vs original BoxFusion.
#
# Arms
#   recar3d_*   : development/reliability_v2 frozen source (2026-09-26 final
#                 method lock) run with its locked config, only data/output
#                 paths rewritten for the prepared 3RScan stream.
#   original_*  : config/scannet.yaml via the main-tree demo (the same
#                 original-BoxFusion arm as the 2026-09-24 example protocol).
#
# Protocol: 3RScan ScanNet18-mapped class-agnostic AABB AP at IoU 0.15/0.25/0.50.

MAIN=/data/ZhaoX/BoxFusion
DEV="$MAIN/development/reliability_v2"
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
RAW_ROOT=/extra/ZhaoX/3RScan
SCAN_LIST="$MAIN/data/3rscan_meta/val_full157.txt"
MAPPING="$MAIN/data/3rscan_meta/3RScan.v2_Semantic_Classes_Mapping.csv"
PREPARED_ROOT="$MAIN/data/3rscan_boxfusion_val157"
RUN_ROOT="$MAIN/reports/3rscan_val157_recar3d_vs_original_20260928"
MODEL="$MAIN/models/cutr_rgbd.pth"
CLIP_MODEL="$MAIN/models/open_clip_pytorch_model.bin"
CLASS_TXT="$MAIN/data/panoptic_categories_nomerge.txt"
LOG_ROOT="$RUN_ROOT/logs"
DRIVER_LOG="$RUN_ROOT/driver.log"
SCENE_TIMEOUT_S=2400

for required in "$PYTHON" "$RAW_ROOT" "$SCAN_LIST" "$MAPPING" "$MODEL" "$CLIP_MODEL" \
  "$CLASS_TXT" "$DEV/config/scannet_t05_boxer_reliability_v2_official100.yaml" \
  "$DEV/boxfusion/reliability_candidate_map.py"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
[[ "$(grep -c . "$SCAN_LIST")" -eq 157 ]] || { echo "Expected 157 validation scans" >&2; exit 1; }

mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$RUN_ROOT/mplconfig" "$RUN_ROOT/failed_attempts"
exec 9>"$RUN_ROOT/run.lock"
flock -n 9 || { echo "Driver already active" >&2; exit 1; }
exec > >(tee -a "$DRIVER_LOG") 2>&1

stamp() { date '+%F %T'; }
echo "[$(stamp)] 3RScan val157 (official full validation split) ReCaR-3D vs original BoxFusion driver started"
sha256sum "$SCAN_LIST" "$MAPPING" "$MAIN/config/scannet.yaml" \
  "$DEV/config/scannet_t05_boxer_reliability_v2_official100.yaml" \
  "$MAIN/tools/prepare_3rscan_boxfusion.py" \
  "$MAIN/tools/build_3rscan_recar3d_configs.py" \
  "$MAIN/tools/materialize_3rscan_recar3d_factorial.py" \
  "$MAIN/tools/eval_3rscan_boxfusion.py" \
  "$DEV/boxfusion/reliability_candidate_map.py" \
  "$DEV/boxfusion/online_candidate_runtime.py" \
  "$DEV/boxfusion/online_candidate_map.py" \
  "$DEV/boxfusion/box_manager.py" "$DEV/boxfusion/instances.py" "$DEV/demo.py" \
  "$MODEL" "$CLIP_MODEL" "$CLASS_TXT" > "$RUN_ROOT/source_fingerprints.sha256"

# ---------------------------------------------------------------- 1. prepare
if [[ ! -s "$PREPARED_ROOT/manifest.json" ]]; then
  echo "[$(stamp)] Preparing 157 validation scans from $RAW_ROOT"
  "$PYTHON" "$MAIN/tools/prepare_3rscan_boxfusion.py" \
    --raw-root "$RAW_ROOT" --output-root "$PREPARED_ROOT" \
    --mapping "$MAPPING" --scan-list "$SCAN_LIST" --alias-start 8000 --resume
fi
for required in "$PREPARED_ROOT/manifest.json" "$PREPARED_ROOT/scenes.txt"; do
  [[ -s "$required" ]] || { echo "Missing prepared input: $required" >&2; exit 1; }
done
mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$PREPARED_ROOT/scenes.txt")
[[ "${#SCENES[@]}" -eq 157 ]] || { echo "Expected 157 prepared scenes, got ${#SCENES[@]}" >&2; exit 1; }
"$PYTHON" - "$PREPARED_ROOT/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
gt = sum(row["gt_scannet18"] for row in m["scans"])
frames = sum(row["frames"] for row in m["scans"])
assert gt > 0, "no ground-truth boxes were mapped; evaluation would be meaningless"
print(f"prepared scenes={len(m['scans'])} frames={frames} gt_boxes={gt}")
PY

# ---------------------------------------------------------------- 2. configs
"$PYTHON" "$MAIN/tools/build_3rscan_recar3d_configs.py" \
  --prepared-root "$PREPARED_ROOT" --run-root "$RUN_ROOT"
RECAR3D_CONFIG="$RUN_ROOT/configs/recar3d_locked.yaml"
ORIGINAL_CONFIG="$RUN_ROOT/configs/original_boxfusion.yaml"

COMPONENTS=(native_native native_first native_mean native_max native_ema \
  native_diverse_max native_reliability births_plr_v1 births_calr_v1 births_calr_v2)

validate_recar3d_scene() {
  local scene="$1" log="$LOG_ROOT/recar3d/$1.log"
  [[ -s "$RUN_ROOT/predictions/recar3d_native/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$RUN_ROOT/predictions/recar3d_online_host/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$RUN_ROOT/diagnostics/recar3d_online/${scene}.json" ]] || return 1
  [[ -s "$RUN_ROOT/components/${scene}.json" ]] || return 1
  [[ -s "$log" ]] || return 1
  [[ "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|RuntimeError|CUDA out of memory' "$log" || return 1
  local component
  for component in "${COMPONENTS[@]}"; do
    [[ -s "$RUN_ROOT/components/$component/${scene}_boxes.pkl" ]] || return 1
  done
  "$PYTHON" - "$RUN_ROOT/diagnostics/recar3d_online/${scene}.json" "$RUN_ROOT/components/${scene}.json" <<'PY' >/dev/null
import json, sys
diagnostic = json.load(open(sys.argv[1], encoding="utf-8"))
components = json.load(open(sys.argv[2], encoding="utf-8"))
assert diagnostic["strictly_causal"] is True
assert diagnostic["online_incremental"] is True
assert diagnostic["scene_end_inference"] is False
assert diagnostic["state"]["cross_branch_dedup"] is False
assert diagnostic["state"]["m1p"]["use_children"] is False
assert components["single_online_pass"] is True
assert components["future_frames_used"] is False
assert len(components["counts"]) == 10
PY
}

validate_original_scene() {
  local scene="$1" log="$LOG_ROOT/original_boxfusion/$1.log"
  [[ -s "$RUN_ROOT/predictions/original_boxfusion/${scene}_boxes.pkl" ]] || return 1
  [[ "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|RuntimeError|CUDA out of memory' "$log" || return 1
}

quarantine_scene() {
  local arm="$1" scene="$2"
  local archive="$RUN_ROOT/failed_attempts/$(date '+%Y%m%d_%H%M%S')_${arm}_${scene}"
  mkdir -p "$archive"
  local path
  if [[ "$arm" == "recar3d" ]]; then
    for path in \
      "$RUN_ROOT/predictions/recar3d_native/${scene}"* \
      "$RUN_ROOT/predictions/recar3d_online_host/${scene}"* \
      "$RUN_ROOT/diagnostics/recar3d_online/${scene}"* \
      "$RUN_ROOT/components/${scene}"* \
      "$RUN_ROOT/components/"*/"${scene}"* \
      "$LOG_ROOT/recar3d/${scene}"*; do
      [[ -e "$path" ]] && mv "$path" "$archive/"
    done
  else
    for path in \
      "$RUN_ROOT/predictions/original_boxfusion/${scene}"* \
      "$LOG_ROOT/original_boxfusion/${scene}"*; do
      [[ -e "$path" ]] && mv "$path" "$archive/"
    done
  fi
  echo "[$(stamp)] Quarantined partial artifacts of $arm/$scene to $archive"
}

# Zero-detection scenes legitimately produce no native pkl: demo.py only
# writes the empty-prediction file when a proposal cache or dynamic branch is
# active, and the locked route uses neither. A completed empty scene is
# repaired post-hoc with the exact save_box([[]]) payload; the route itself is
# untouched.
repair_recar3d_empty_scene() {
  local scene="$1" log="$LOG_ROOT/recar3d/$1.log"
  [[ "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|RuntimeError|CUDA out of memory' "$log" || return 1
  [[ -s "$RUN_ROOT/components/${scene}.json" ]] || return 1
  "$PYTHON" - "$RUN_ROOT" "$scene" <<'PY'
import json, os, pickle, sys

root, scene = sys.argv[1], sys.argv[2]
counts = json.load(open(f"{root}/components/{scene}.json", encoding="utf-8"))["counts"]
assert counts, "empty component manifest"
assert all(value == 0 for value in counts.values()), f"non-empty scene: {counts}"
path = f"{root}/predictions/recar3d_native/{scene}_boxes.pkl"
assert not os.path.exists(path), f"native pkl unexpectedly exists: {path}"
with open(path, "wb") as handle:
    pickle.dump([[]], handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.flush()
    os.fsync(handle.fileno())
print(f"repaired zero-detection scene {scene} with empty native pkl")
PY
}

repair_original_empty_scene() {
  local scene="$1" log="$LOG_ROOT/original_boxfusion/$1.log"
  [[ "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|RuntimeError|CUDA out of memory' "$log" || return 1
  [[ ! -e "$RUN_ROOT/predictions/original_boxfusion/${scene}_boxes.pkl" ]] || return 1
  "$PYTHON" - "$RUN_ROOT" "$scene" <<'PY'
import os, pickle, sys

root, scene = sys.argv[1], sys.argv[2]
path = f"{root}/predictions/original_boxfusion/{scene}_boxes.pkl"
with open(path, "wb") as handle:
    pickle.dump([[]], handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.flush()
    os.fsync(handle.fileno())
print(f"repaired zero-detection scene {scene} with empty original pkl")
PY
}

run_arm_scene() {
  local arm="$1" tree="$2" config="$3" scene="$4" log="$5" gpu_mask="$6"
  echo "[$(stamp)] Running $arm $scene"
  (
    cd "$tree"
    CUDA_VISIBLE_DEVICES="$gpu_mask" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
    MPLCONFIGDIR="$RUN_ROOT/mplconfig" \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
    PYTHONPATH="$tree:$MAIN/third_party/WeDetect" \
    timeout "$SCENE_TIMEOUT_S" "$PYTHON" demo.py scannet --model-path "$MODEL" \
      --clip_path "$CLIP_MODEL" --class_txt "$CLASS_TXT" --config "$config" \
      --device cuda:0 --seq "$scene"
  ) >"$log" 2>&1 || return 1
}

# ------------------------------------------------------- 3. ReCaR-3D arm
for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"
  if validate_recar3d_scene "$scene" \
     || { repair_recar3d_empty_scene "$scene" && validate_recar3d_scene "$scene"; }; then
    echo "[$(stamp)] [$((index+1))/157] Reusing recar3d $scene"; continue
  fi
  if ! run_arm_scene recar3d "$DEV" "$RECAR3D_CONFIG" "$scene" "$LOG_ROOT/recar3d/$scene.log" "0,1"; then
    echo "[$(stamp)] recar3d $scene failed; quarantining and retrying once"
    quarantine_scene recar3d "$scene"
    run_arm_scene recar3d "$DEV" "$RECAR3D_CONFIG" "$scene" "$LOG_ROOT/recar3d/$scene.log" "0,1" || {
      echo "[$(stamp)] recar3d $scene failed twice; see $LOG_ROOT/recar3d/$scene.log" >&2; exit 1;
    }
  fi
  validate_recar3d_scene "$scene" \
     || { repair_recar3d_empty_scene "$scene" && validate_recar3d_scene "$scene"; } \
     || { echo "[$(stamp)] recar3d $scene produced invalid artifacts" >&2; exit 1; }
  echo "[$(stamp)] [$((index+1))/157] Completed recar3d $scene"
done

# ------------------------------------------------------- 4. original arm
for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"
  if validate_original_scene "$scene" \
     || { repair_original_empty_scene "$scene" && validate_original_scene "$scene"; }; then
    echo "[$(stamp)] [$((index+1))/157] Reusing original $scene"; continue
  fi
  if ! run_arm_scene original_boxfusion "$MAIN" "$ORIGINAL_CONFIG" "$scene" "$LOG_ROOT/original_boxfusion/$scene.log" "0"; then
    echo "[$(stamp)] original $scene failed; quarantining and retrying once"
    quarantine_scene original_boxfusion "$scene"
    run_arm_scene original_boxfusion "$MAIN" "$ORIGINAL_CONFIG" "$scene" "$LOG_ROOT/original_boxfusion/$scene.log" "0" || {
      echo "[$(stamp)] original $scene failed twice; see $LOG_ROOT/original_boxfusion/$scene.log" >&2; exit 1;
    }
  fi
  validate_original_scene "$scene" \
     || { repair_original_empty_scene "$scene" && validate_original_scene "$scene"; } \
     || { echo "[$(stamp)] original $scene produced invalid artifacts" >&2; exit 1; }
  echo "[$(stamp)] [$((index+1))/157] Completed original $scene"
done

# ------------------------------------------- 5. factorial assembly + eval
FACTORIAL_ROOT="$RUN_ROOT/predictions/factorial"
if [[ ! -s "$FACTORIAL_ROOT/manifest.json" ]]; then
  "$PYTHON" "$MAIN/tools/materialize_3rscan_recar3d_factorial.py" \
    --scene-list "$PREPARED_ROOT/scenes.txt" \
    --components "$RUN_ROOT/components" --output "$FACTORIAL_ROOT"
fi
"$PYTHON" "$MAIN/tools/eval_3rscan_boxfusion.py" \
  --prepared-root "$PREPARED_ROOT" \
  --prediction original_boxfusion "$RUN_ROOT/predictions/original_boxfusion" \
  --prediction recar3d_base "$FACTORIAL_ROOT/base" \
  --prediction recar3d_plr "$FACTORIAL_ROOT/plr" \
  --prediction recar3d_calr "$FACTORIAL_ROOT/calr" \
  --prediction recar3d_mvsr "$FACTORIAL_ROOT/mvsr" \
  --prediction recar3d_plr_calr "$FACTORIAL_ROOT/plr_calr" \
  --prediction recar3d_plr_mvsr "$FACTORIAL_ROOT/plr_mvsr" \
  --prediction recar3d_calr_mvsr "$FACTORIAL_ROOT/calr_mvsr" \
  --prediction recar3d_full "$FACTORIAL_ROOT/full" \
  --output "$RUN_ROOT/report"

"$PYTHON" - "$RUN_ROOT" "$PREPARED_ROOT" <<'PY'
import json, sys
from pathlib import Path

run_root, prepared_root = Path(sys.argv[1]), Path(sys.argv[2])
metrics = json.loads((run_root / "report" / "metrics.json").read_text(encoding="utf-8"))
manifest = json.loads((prepared_root / "manifest.json").read_text(encoding="utf-8"))
order = ["original_boxfusion", "recar3d_base", "recar3d_plr", "recar3d_calr",
         "recar3d_mvsr", "recar3d_plr_calr", "recar3d_plr_mvsr",
         "recar3d_calr_mvsr", "recar3d_full"]
lines = [
    "# 3RScan val157: locked ReCaR-3D vs original BoxFusion (official full validation split)",
    "",
    f"Protocol: `{metrics['protocol']}`; scenes: {metrics['scenes']}; "
    f"GT boxes: {metrics['ground_truth_boxes']}; "
    f"AP at IoU 0.15/0.25/0.50.",
    "",
    "- original arm: `config/scannet.yaml` via the main-tree demo (same protocol as the 2026-09-24 example).",
    "- recar3d arms: frozen `development/reliability_v2` source (2026-09-26 final method lock), "
    "locked config with only data/output paths rewritten; factorial assembled from one online pass "
    "with terminal row order native -> PLR -> CALR.",
    "",
    "| Arm | AP15 | AP25 | AP50 | Predictions |",
    "|---|---:|---:|---:|---:|",
]
for name in order:
    arm = metrics["arms"][name]
    values = [arm["metrics"][str(t)]["ap"] for t in (0.15, 0.25, 0.5)]
    lines.append(f"| {name} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | {arm['prediction_boxes']} |")
frames = sum(row["frames"] for row in manifest["scans"])
lines += [
    "",
    f"Prepared scan summary: {len(manifest['scans'])} scans, {frames} frames total "
    f"(scene list: `data/3rscan_meta/val_full157.txt`, alias-start 8000).",
    "",
    "Provenance: `source_fingerprints.sha256`, `configs/`, `components/`, `report/metrics.json`.",
]
(run_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {run_root / 'REPORT.md'}")
PY

echo "[$(stamp)] driver complete: $RUN_ROOT/REPORT.md"
