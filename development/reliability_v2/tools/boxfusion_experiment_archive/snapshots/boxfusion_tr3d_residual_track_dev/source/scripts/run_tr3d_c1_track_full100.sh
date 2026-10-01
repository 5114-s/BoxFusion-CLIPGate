#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

TAG="${1:-c1_r3active_full100_v1}"
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$ ]] || {
  echo "usage: $0 [unique-run-tag]" >&2
  exit 2
}

PYTHON_BIN="${BOXFUSION_C1_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
R2_ROOT="${BOXFUSION_C1_R2_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev}"
INPUT_ROOT="${BOXFUSION_C1_INPUT_ROOT:-$ROOT/artifacts/tr3d_c2_full100_inputs/v1}"
PARENT_ROOT="${BOXFUSION_C1_PARENT_CACHE_ROOT:-$R2_ROOT/cache/tr3d_prefix_boxfusion_causal_p100_full100_v3}"
R2A_ROOT="${BOXFUSION_C1_R2A_CACHE_ROOT:-$INPUT_ROOT/r2a_cache}"
R2B_ROOT="${BOXFUSION_C1_R2B_CACHE_ROOT:-$INPUT_ROOT/r2b_cache}"
R2A_REPORT="${BOXFUSION_C1_R2A_EXPORT_REPORT:-$INPUT_ROOT/r2a_reports/export_report.json}"
R2B_REPORT="${BOXFUSION_C1_R2B_EXPORT_REPORT:-$INPUT_ROOT/r2b_reports/export_report.json}"
SCENE_LIST="${BOXFUSION_C1_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
SCANS_ROOT="${BOXFUSION_C1_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
GT_ROOT="${BOXFUSION_C1_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
ACTIVE_ROOT="${BOXFUSION_C1_ACTIVE_ROOT:-/extra/ZhaoX/codex_artifacts/boxfusion_r3_20260805/boxfusion_tr3d_terminal_paired_full100/results/b6_g0_tr3d_terminal/g0_tr3d_terminal_paired_full100_v1/scannetv2_val-4b18fc586f7a}"
PARENT_CHECKPOINT_SHA="${BOXFUSION_C1_PARENT_CHECKPOINT_SHA256:-a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448}"
PARENT_CONFIG_SHA="${BOXFUSION_C1_PARENT_CONFIG_SHA256:-709b66d9e244ef4385dfa9bbc89895ad06c78534f9d14bb7149b687fd58da785}"
RUN_ROOT="${BOXFUSION_C1_ARTIFACT_ROOT:-$ROOT/artifacts/tr3d_c1_track}/$TAG"
CACHE_ROOT="$RUN_ROOT/cache"
REPORT_ROOT="$RUN_ROOT/reports"

for path in "$PYTHON_BIN" "$R2A_REPORT" "$R2B_REPORT" "$SCENE_LIST"; do
  [[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
for path in "$PARENT_ROOT" "$R2A_ROOT" "$R2B_ROOT" "$ACTIVE_ROOT" "$SCANS_ROOT" "$GT_ROOT"; do
  [[ -d "$path" ]] || { echo "Missing required directory: $path" >&2; exit 2; }
done
scene_count="$(awk 'NF && $1 !~ /^#/ {count += 1} END {print count + 0}' "$SCENE_LIST")"
[[ "$scene_count" == "100" ]] || { echo "C1 full100 requires exactly 100 scenes" >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing immutable run overwrite: $RUN_ROOT" >&2; exit 2; }
mkdir -p "$CACHE_ROOT" "$REPORT_ROOT"

echo "C1 unmatched TR3D multi-view evidence-track observer (full100)"
echo "  frozen anchor: R3 active"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  cache reuse: TR3D p100 + R2a depth/free-space + R2b DINO"
echo "  output: $RUN_ROOT"
echo "  contract: observer_only=true; applied_count=0; CLIP/predictions unchanged"

env -u PYTHONPATH TMPDIR="${BOXFUSION_C1_TMPDIR:-/dev/shm}" PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON_BIN" tools/run_tr3d_c1_track_observer.py \
    --parent-cache-root "$PARENT_ROOT" \
    --r2a-cache-root "$R2A_ROOT" --r2b-cache-root "$R2B_ROOT" \
    --r2a-export-report "$R2A_REPORT" --r2b-export-report "$R2B_REPORT" \
    --active-prediction-root "$ACTIVE_ROOT" --scene-list "$SCENE_LIST" \
    --scans-root "$SCANS_ROOT" --output-root "$CACHE_ROOT" --prefix-id p100 \
    --expected-parent-checkpoint-sha256 "$PARENT_CHECKPOINT_SHA" \
    --expected-parent-config-sha256 "$PARENT_CONFIG_SHA" \
    --report "$REPORT_ROOT/export_report.json" \
    > "$REPORT_ROOT/export_stdout.json"

env -u PYTHONPATH TMPDIR="${BOXFUSION_C1_TMPDIR:-/dev/shm}" PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON_BIN" tools/audit_tr3d_c1_track_observer.py \
    --export-report "$REPORT_ROOT/export_report.json" \
    --c1-cache-root "$CACHE_ROOT" --parent-cache-root "$PARENT_ROOT" \
    --active-prediction-root "$ACTIVE_ROOT" --scene-list "$SCENE_LIST" \
    --prefix-id p100 --gt-root "$GT_ROOT" --scans-root "$SCANS_ROOT" \
    --report "$REPORT_ROOT/gt_audit.json" \
    > "$REPORT_ROOT/audit_stdout.json"

"$PYTHON_BIN" tools/summarize_tr3d_c1_track.py "$REPORT_ROOT/gt_audit.json" \
  | tee "$REPORT_ROOT/summary.txt"
echo "C1 full100 complete: $REPORT_ROOT/gt_audit.json"
